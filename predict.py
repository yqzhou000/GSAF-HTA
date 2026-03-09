import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
from datetime import datetime
from rdkit import Chem


# ==================== 数据编码器 ====================

class SMILESEncoder:
    """SMILES字符串编码器"""
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.char_set = set("CNOSPFIBrClcnos@=#()[]+-./\\1234567890")
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(self.char_set))}
        self.char_to_idx['<PAD>'] = 0
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, smiles):
        encoded = []
        for char in smiles[:self.max_length]:
            encoded.append(self.char_to_idx.get(char, 0))
        while len(encoded) < self.max_length:
            encoded.append(0)
        return np.array(encoded, dtype=np.int64)


class ProteinSeqEncoder:
    """蛋白质序列编码器"""
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['<PAD>'] = 0
        self.aa_to_idx['X'] = len(self.aa_to_idx)
        self.vocab_size = len(self.aa_to_idx)
    
    def encode(self, sequence):
        encoded = []
        for aa in sequence[:self.max_length]:
            encoded.append(self.aa_to_idx.get(aa, self.aa_to_idx['X']))
        while len(encoded) < self.max_length:
            encoded.append(0)
        return np.array(encoded, dtype=np.int64)


class MolecularGraphBuilder:
    def __init__(self, max_atoms=100):
        self.max_atoms = max_atoms
        self.atom_types = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'At', 'H'
        ]
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}
        self.num_atom_types = len(self.atom_types) + 1  # UNK

    def build(self, smiles):
        node_feats = np.zeros((self.max_atoms, self.num_atom_types), dtype=np.float32)
        adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
        mask = np.zeros((self.max_atoms,), dtype=np.float32)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return node_feats, adj, mask

        num_atoms = min(mol.GetNumAtoms(), self.max_atoms)
        for i in range(num_atoms):
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            idx = self.atom_to_idx.get(symbol, self.num_atom_types - 1)
            node_feats[i, idx] = 1.0
            mask[i] = 1.0

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        for i in range(num_atoms):
            adj[i, i] = 1.0

        return node_feats, adj, mask


class DTAPredictDataset(Dataset):
    """DTA预测数据集"""
    def __init__(self, smiles_list, sequence_list, smiles_encoder, protein_encoder, graph_builder):
        self.smiles_list = smiles_list
        self.sequence_list = sequence_list
        self.smiles_encoder = smiles_encoder
        self.protein_encoder = protein_encoder
        self.graph_builder = graph_builder
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_encoder.encode(self.smiles_list[idx])
        protein = self.protein_encoder.encode(self.sequence_list[idx])
        node_feats, adj, mask = self.graph_builder.build(self.smiles_list[idx])
        
        return {
            'smiles': torch.LongTensor(smiles),
            'protein': torch.LongTensor(protein),
            'graph_x': torch.FloatTensor(node_feats),
            'graph_adj': torch.FloatTensor(adj),
            'graph_mask': torch.FloatTensor(mask),
            'index': idx
        }


# ==================== 模型组件 ====================

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, node_feats, adj, mask):
        deg = adj.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        h = node_feats
        for layer in self.layers:
            h = torch.matmul(norm_adj, h)
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)

        attn_logits = self.attention(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * h, dim=1)
        return pooled


class DrugEncoder(nn.Module):
    """药物编码器"""
    def __init__(self, vocab_size, graph_feat_dim, embedding_dim=128, num_filters=64, graph_hidden=64):
        super(DrugEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度CNN
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        # 注意力
        self.attention = nn.Linear(num_filters * 3, 1)
        self.dropout = nn.Dropout(0.3)
        self.graph_encoder = GraphEncoder(graph_feat_dim, hidden_dim=graph_hidden)
        self.output_dim = num_filters * 3 + self.graph_encoder.output_dim
    
    def forward(self, x, graph_x, graph_adj, graph_mask):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        
        attention_weights = torch.softmax(self.attention(conv_out), dim=1)
        attended = (conv_out * attention_weights).sum(dim=1)
        
        seq_features = self.dropout(attended)
        graph_features = self.graph_encoder(graph_x, graph_adj, graph_mask)
        return torch.cat([seq_features, graph_features], dim=1)


class ProteinEncoderModel(nn.Module):
    """蛋白质编码器"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=64, lstm_hidden=128):
        super(ProteinEncoderModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度CNN
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            num_filters * 3,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 注意力
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.output_dim = lstm_hidden * 2
    
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        
        lstm_out, _ = self.lstm(conv_out)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = (lstm_out * attention_weights).sum(dim=1)
        
        return self.dropout(attended)


class DTAPredictor(nn.Module):
    """DTA预测模型"""
    def __init__(self, drug_vocab_size, protein_vocab_size, graph_feat_dim):
        super(DTAPredictor, self).__init__()
        
        self.drug_encoder = DrugEncoder(
            drug_vocab_size,
            graph_feat_dim=graph_feat_dim,
            embedding_dim=128,
            num_filters=64
        )
        self.protein_encoder = ProteinEncoderModel(protein_vocab_size, embedding_dim=128, 
                                                   num_filters=64, lstm_hidden=128)
        
        combined_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim

        self.cross_attention = nn.MultiheadAttention(
            combined_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc_out = nn.Linear(128, 1)
    
    def forward(self, drug, protein, graph_x, graph_adj, graph_mask):
        drug_features = self.drug_encoder(drug, graph_x, graph_adj, graph_mask)
        protein_features = self.protein_encoder(protein)
        
        combined = torch.cat([drug_features, protein_features], dim=1)
        combined_seq = combined.unsqueeze(1)
        attended, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(1)
        enhanced = combined + attended
        
        x = F.relu(self.bn1(self.fc1(enhanced)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        output = self.fc_out(x)
        
        return output


# ==================== 预测函数 ====================

def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"🔄 加载模型: {model_path}")
    
    # 初始化编码器
    smiles_encoder = SMILESEncoder(max_length=100)
    protein_encoder = ProteinSeqEncoder(max_length=1000)
    graph_builder = MolecularGraphBuilder(max_atoms=100)
    
    # 初始化模型
    model = DTAPredictor(
        drug_vocab_size=smiles_encoder.vocab_size,
        protein_vocab_size=protein_encoder.vocab_size,
        graph_feat_dim=graph_builder.num_atom_types
    ).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("✓ 模型加载成功\n")
    
    return model, smiles_encoder, protein_encoder, graph_builder


def predict_single(model, smiles, sequence, smiles_encoder, protein_encoder, graph_builder, device='cuda'):
    """预测单个药物-靶标对的亲和力"""
    
    # 编码
    drug_encoded = torch.LongTensor(smiles_encoder.encode(smiles)).unsqueeze(0).to(device)
    protein_encoded = torch.LongTensor(protein_encoder.encode(sequence)).unsqueeze(0).to(device)
    graph_x, graph_adj, graph_mask = graph_builder.build(smiles)
    graph_x = torch.FloatTensor(graph_x).unsqueeze(0).to(device)
    graph_adj = torch.FloatTensor(graph_adj).unsqueeze(0).to(device)
    graph_mask = torch.FloatTensor(graph_mask).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        prediction = model(drug_encoded, protein_encoded, graph_x, graph_adj, graph_mask)
    
    return prediction.item()


def predict_batch(model, smiles_list, sequence_list, smiles_encoder, protein_encoder, graph_builder, 
                 device='cuda', batch_size=128):
    """批量预测"""
    
    print(f"📊 批量预测中...")
    print(f"   样本数: {len(smiles_list)}")
    print(f"   Batch大小: {batch_size}\n")
    
    # 创建数据集
    dataset = DTAPredictDataset(smiles_list, sequence_list, smiles_encoder, protein_encoder, graph_builder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            drug = batch['smiles'].to(device)
            protein = batch['protein'].to(device)
            graph_x = batch['graph_x'].to(device)
            graph_adj = batch['graph_adj'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            
            pred = model(drug, protein, graph_x, graph_adj, graph_mask)
            predictions.extend(pred.cpu().numpy().flatten().tolist())
    
    print("✓ 预测完成\n")
    
    return predictions


def save_predictions(smiles_list, sequence_list, predictions, output_path, 
                    true_values=None, drug_names=None, target_names=None):
    """保存预测结果到CSV"""
    
    results = {
        'SMILES': smiles_list,
        'Sequence': sequence_list,
        'Predicted_pIC50': predictions
    }
    
    # 添加可选列
    if drug_names is not None:
        results['Drug_Name'] = drug_names
    
    if target_names is not None:
        results['Target_Name'] = target_names
    
    if true_values is not None:
        results['True_pIC50'] = true_values
        results['Error'] = [pred - true for pred, true in zip(predictions, true_values)]
        results['Absolute_Error'] = [abs(pred - true) for pred, true in zip(predictions, true_values)]
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 重新排列列顺序
    columns_order = []
    if 'Drug_Name' in df.columns:
        columns_order.append('Drug_Name')
    if 'Target_Name' in df.columns:
        columns_order.append('Target_Name')
    columns_order.extend(['SMILES', 'Sequence', 'Predicted_pIC50'])
    if 'True_pIC50' in df.columns:
        columns_order.extend(['True_pIC50', 'Error', 'Absolute_Error'])
    
    df = df[columns_order]
    
    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"💾 结果已保存到: {output_path}")
    
    # 显示统计信息
    if true_values is not None:
        mae = df['Absolute_Error'].mean()
        rmse = np.sqrt((df['Error'] ** 2).mean())
        print(f"\n统计信息:")
        print(f"   MAE (平均绝对误差): {mae:.4f}")
        print(f"   RMSE (均方根误差): {rmse:.4f}")
    
    return df


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='DTA预测脚本')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径 (例如: /root/1/result/best_dta_model_stage2.pt)')
    parser.add_argument('--input', type=str, required=True,
                       help='输入文件路径 (CSV或Excel, 必须包含Smiles和Sequence列)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件路径 (CSV格式)')
    
    # 可选参数
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='使用设备 (默认: cuda)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批处理大小 (默认: 128)')
    parser.add_argument('--smiles_col', type=str, default='Smiles',
                       help='SMILES列名 (默认: Smiles)')
    parser.add_argument('--sequence_col', type=str, default='Sequence',
                       help='蛋白质序列列名 (默认: Sequence)')
    parser.add_argument('--true_col', type=str, default=None,
                       help='真实值列名 (可选, 用于计算误差)')
    parser.add_argument('--drug_name_col', type=str, default=None,
                       help='药物名称列名 (可选)')
    parser.add_argument('--target_name_col', type=str, default=None,
                       help='靶标名称列名 (可选)')
    
    args = parser.parse_args()
    
    # 打印配置
    print("\n" + "="*70)
    print(" "*20 + "DTA亲和力预测")
    print("="*70)
    print(f"模型文件: {args.model}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"设备: {args.device}")
    print(f"Batch大小: {args.batch_size}")
    print("="*70 + "\n")
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # 加载模型
    model, smiles_encoder, protein_encoder, graph_builder = load_model(args.model, device)
    
    # 读取输入数据
    print(f"📖 读取输入文件: {args.input}")
    if args.input.endswith('.xlsx') or args.input.endswith('.xls'):
        df = pd.read_excel(args.input)
    elif args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
    
    print(f"✓ 读取 {len(df)} 条数据\n")
    
    # 检查必需列
    if args.smiles_col not in df.columns:
        raise ValueError(f"找不到SMILES列: {args.smiles_col}")
    if args.sequence_col not in df.columns:
        raise ValueError(f"找不到Sequence列: {args.sequence_col}")
    
    # 提取数据
    smiles_list = df[args.smiles_col].tolist()
    sequence_list = df[args.sequence_col].tolist()
    
    true_values = None
    if args.true_col and args.true_col in df.columns:
        true_values = df[args.true_col].tolist()
        print(f"✓ 包含真实值列: {args.true_col}\n")
    
    drug_names = None
    if args.drug_name_col and args.drug_name_col in df.columns:
        drug_names = df[args.drug_name_col].tolist()
    
    target_names = None
    if args.target_name_col and args.target_name_col in df.columns:
        target_names = df[args.target_name_col].tolist()
    
    # 批量预测
    predictions = predict_batch(
        model, smiles_list, sequence_list, 
        smiles_encoder, protein_encoder, graph_builder,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 保存结果
    results_df = save_predictions(
        smiles_list, sequence_list, predictions, args.output,
        true_values=true_values,
        drug_names=drug_names,
        target_names=target_names
    )
    
    # 显示前几条预测结果
    print("\n前5条预测结果:")
    print("-"*70)
    print(results_df.head())
    print("-"*70)
    
    print("\n预测完成! 🎉\n")


def predict_interactive():
    """交互式预测模式"""
    print("\n" + "="*70)
    print(" "*20 + "DTA亲和力预测 - 交互模式")
    print("="*70 + "\n")
    
    # 输入模型路径
    model_path = input("请输入模型文件路径: ").strip()
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 加载模型
    model, smiles_encoder, protein_encoder, graph_builder = load_model(model_path, device)
    
    while True:
        print("\n" + "-"*70)
        smiles = input("请输入SMILES字符串 (输入q退出): ").strip()
        
        if smiles.lower() == 'q':
            print("退出预测")
            break
        
        sequence = input("请输入蛋白质序列: ").strip()
        
        if not smiles or not sequence:
            print("⚠️  输入不能为空")
            continue
        
        # 预测
        try:
            prediction = predict_single(
                model, smiles, sequence,
                smiles_encoder, protein_encoder, graph_builder, device
            )
            
            print(f"\n预测结果:")
            print(f"   pIC50: {prediction:.4f}")
            print(f"   IC50: {10**(-prediction):.2e} M")
            
        except Exception as e:
            print(f"⚠️  预测失败: {str(e)}")


if __name__ == "__main__":
    import sys
    
    # 如果没有命令行参数，启动交互模式
    if len(sys.argv) == 1:
        predict_interactive()
    else:
        main()