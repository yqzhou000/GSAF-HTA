import pandas as pd
import numpy as np
from predict import load_model, predict_single, predict_batch, save_predictions


def example_1_single_prediction():
    """示例1: 单个药物-靶标对预测"""
    print("\n" + "="*70)
    print(" "*20 + "示例1: 单个预测")
    print("="*70 + "\n")
    
    # 加载模型
    model_path = '/root/1/result/best_dta_model_stage2.pt'
    device = 'cuda'
    
    model, smiles_encoder, protein_encoder = load_model(model_path, device)
    
    # 示例数据 - 伊布洛芬 (Ibuprofen) 与 COX-2
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    sequence = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF"
    
    # 预测
    prediction = predict_single(
        model, smiles, sequence,
        smiles_encoder, protein_encoder,
        device=device
    )
    
    # 输出结果
    ic50 = 10**(-prediction)
    
    print("药物: Ibuprofen")
    print("靶标: COX-2")
    print(f"SMILES: {smiles[:50]}...")
    print(f"\n预测结果:")
    print(f"  pIC50: {prediction:.4f}")
    print(f"  IC50:  {ic50:.2e} M")
    
    # 解释
    if prediction >= 8:
        potency = "非常强效"
    elif prediction >= 7:
        potency = "强效"
    elif prediction >= 6:
        potency = "中等效力"
    else:
        potency = "弱效"
    
    print(f"  效力评估: {potency}")


def example_2_batch_prediction():
    """示例2: 批量预测"""
    print("\n" + "="*70)
    print(" "*20 + "示例2: 批量预测")
    print("="*70 + "\n")
    
    # 加载模型
    model_path = '/root/1/result/best_dta_model_stage2.pt'
    device = 'cuda'
    
    model, smiles_encoder, protein_encoder = load_model(model_path, device)
    
    # 示例数据 - 多个药物
    data = {
        'Drug_Name': ['Ibuprofen', 'Celecoxib', 'Aspirin'],
        'Smiles': [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
            'COc1ccc2nc(sc2c1)S(=O)(=O)N',
            'CC(=O)Oc1ccccc1C(=O)O'
        ],
        'Sequence': [
            'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF',
            'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF',
            'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF'
        ],
        'Target_Name': ['COX-2', 'COX-2', 'COX-1']
    }
    
    # 批量预测
    predictions = predict_batch(
        model,
        data['Smiles'],
        data['Sequence'],
        smiles_encoder,
        protein_encoder,
        device=device,
        batch_size=128
    )
    
    # 添加预测结果
    data['Predicted_pIC50'] = predictions
    data['Predicted_IC50_M'] = [10**(-p) for p in predictions]
    
    # 显示结果
    df = pd.DataFrame(data)
    print("预测结果:")
    print("-"*70)
    for idx, row in df.iterrows():
        print(f"\n{idx+1}. {row['Drug_Name']} vs {row['Target_Name']}")
        print(f"   pIC50: {row['Predicted_pIC50']:.4f}")
        print(f"   IC50:  {row['Predicted_IC50_M']:.2e} M")
    print("\n" + "-"*70)
    
    # 保存到文件
    output_path = '/root/1/predictions/example_batch_predictions.csv'
    df[['Drug_Name', 'Target_Name', 'Predicted_pIC50', 'Predicted_IC50_M']].to_csv(
        output_path, index=False, float_format='%.4f'
    )
    print(f"\n结果已保存到: {output_path}")


def example_3_compare_models():
    """示例3: 比较不同模型的预测结果"""
    print("\n" + "="*70)
    print(" "*20 + "示例3: 比较不同模型")
    print("="*70 + "\n")
    
    # 测试数据
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    sequence = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF"
    
    device = 'cuda'
    
    # 不同模型
    models = {
        'Stage1_BestCI': '/root/1/result/best_dta_model_bestCI.pt',
        'Stage1_BestRm2': '/root/1/result/best_dta_model_bestRm2.pt',
        'Stage2_Final': '/root/1/result/best_dta_model_stage2.pt'
    }
    
    results = []
    
    print("对同一药物-靶标对使用不同模型进行预测:\n")
    print("药物: Ibuprofen")
    print("靶标: COX-2")
    print("\n" + "-"*70)
    
    for model_name, model_path in models.items():
        try:
            # 加载模型
            model, smiles_encoder, protein_encoder = load_model(model_path, device)
            
            # 预测
            prediction = predict_single(
                model, smiles, sequence,
                smiles_encoder, protein_encoder,
                device=device
            )
            
            results.append({
                'Model': model_name,
                'pIC50': prediction,
                'IC50_M': 10**(-prediction)
            })
            
            print(f"\n{model_name}:")
            print(f"  pIC50: {prediction:.4f}")
            print(f"  IC50:  {10**(-prediction):.2e} M")
            
        except FileNotFoundError:
            print(f"\n⚠️  {model_name}: 模型文件未找到")
            continue
    
    print("\n" + "-"*70)
    
    # 计算一致性
    if len(results) > 1:
        predictions = [r['pIC50'] for r in results]
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        print(f"\n模型预测一致性:")
        print(f"  平均值: {mean_pred:.4f}")
        print(f"  标准差: {std_pred:.4f}")
        print(f"  变异系数: {(std_pred/mean_pred*100):.2f}%")


def example_4_with_true_values():
    """示例4: 带真实值的预测评估"""
    print("\n" + "="*70)
    print(" "*20 + "示例4: 预测评估")
    print("="*70 + "\n")
    
    # 加载模型
    model_path = '/root/1/result/best_dta_model_stage2.pt'
    device = 'cuda'
    
    model, smiles_encoder, protein_encoder = load_model(model_path, device)
    
    # 示例数据 (假设的真实值)
    data = {
        'Drug_Name': ['Ibuprofen', 'Celecoxib', 'Aspirin'],
        'Smiles': [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
            'COc1ccc2nc(sc2c1)S(=O)(=O)N',
            'CC(=O)Oc1ccccc1C(=O)O'
        ],
        'Sequence': [
            'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF',
            'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF',
            'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQPLLILQGLF'
        ],
        'True_pIC50': [7.5, 8.2, 6.8]  # 假设的真实值
    }
    
    # 批量预测
    predictions = predict_batch(
        model,
        data['Smiles'],
        data['Sequence'],
        smiles_encoder,
        protein_encoder,
        device=device,
        batch_size=128
    )
    
    # 保存结果
    output_path = '/root/1/predictions/example_with_evaluation.csv'
    results_df = save_predictions(
        data['Smiles'],
        data['Sequence'],
        predictions,
        output_path,
        true_values=data['True_pIC50'],
        drug_names=data['Drug_Name']
    )
    
    # 详细显示
    print("\n预测结果对比:")
    print("-"*70)
    for idx, row in results_df.iterrows():
        print(f"\n{idx+1}. {row['Drug_Name']}")
        print(f"   真实值:   {row['True_pIC50']:.4f}")
        print(f"   预测值:   {row['Predicted_pIC50']:.4f}")
        print(f"   误差:     {row['Error']:.4f}")
        print(f"   绝对误差: {row['Absolute_Error']:.4f}")
    print("\n" + "-"*70)


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print(" "*15 + "DTA预测 - Python API 示例集")
    print("="*70)
    
    try:
        # 示例1: 单个预测
        example_1_single_prediction()
        
        # 示例2: 批量预测
        example_2_batch_prediction()
        
        # 示例3: 比较模型
        example_3_compare_models()
        
        # 示例4: 带真实值评估
        example_4_with_true_values()
        
        print("\n" + "="*70)
        print(" "*20 + "所有示例运行完成!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()