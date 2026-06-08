import json
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix
    )

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_trainloss(EPOCHS, history, output_dir):
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history['train_loss'], 'b-', linewidth=1.5, label='训练 Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history['train_acc'], 'b-', linewidth=1.5, label='训练准确率')
    ax2.plot(epochs_range, history['val_acc'], 'r-', linewidth=1.5, label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练 & 验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 损失曲线 → {loss_curve_path}")

def draw_cm(all_labels=None, all_preds=None, classes=None, output_dir=None, task=None):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    if task == 'digits':
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(classes, fontsize=12)
        ax.set_yticklabels(classes, fontsize=12)
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title('数字识别 — 混淆矩阵 (归一化)')

        for i in range(10):
            for j in range(10):
                val = cm_normalized[i, j]
                if val > 0.03:
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=10, color=color)


    elif task == 'letters':
        fig, ax = plt.subplots(figsize=(16, 14))
        im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(26))
        ax.set_yticks(range(26))
        ax.set_xticklabels(classes, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title('字母识别 — 混淆矩阵 (归一化)')

        for i in range(26):
            for j in range(26):
                val = cm_normalized[i, j]
                if val > 0.05:
                    color = 'white' if val > 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('比例')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 混淆矩阵 → {cm_path}")

def draw_per_class_metric(all_labels=None, all_preds=None, classes=None, output_dir=None, task=None):
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    overall_accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    if task == 'digits':
        x = np.arange(10)
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))
        bars1 = ax.bar(x - width, per_class_precision, width, label='Precision', color='#2ecc71')
        bars2 = ax.bar(x, per_class_recall, width, label='Recall', color='#3498db')
        bars3 = ax.bar(x + width, per_class_f1, width, label='F1 Score', color='#e74c3c')

        ax.set_xlabel('数字类别')
        ax.set_ylabel('分数')
        ax.set_title('每类分类指标 (Precision / Recall / F1)')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=12)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.3)

    if task == 'letters':
        x = np.arange(26)
        width = 0.25

        fig, ax = plt.subplots(figsize=(18, 8))
        bars1 = ax.bar(x - width, per_class_precision, width, label='Precision', color='#2ecc71')
        bars2 = ax.bar(x, per_class_recall, width, label='Recall', color='#3498db')
        bars3 = ax.bar(x + width, per_class_f1, width, label='F1 Score', color='#e74c3c')

        ax.set_xlabel('字母类别')
        ax.set_ylabel('分数')
        ax.set_title('每类分类指标 (Precision / Recall / F1)')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=9)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.3)

    # 标注总体指标
    summary_text = (f'Overall Accuracy: {overall_accuracy:.4f}\n'
                    f'Macro Precision:  {macro_precision:.4f}\n'
                    f'Macro Recall:     {macro_recall:.4f}\n'
                    f'Macro F1:         {macro_f1:.4f}')
    ax.text(0.98, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
    fig.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 每类指标图 → {metrics_path}")
    per_class_metrics = {}
    for i, cls in enumerate(classes):
        per_class_metrics[cls] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i])
        }
    metrics_json = {
        'overall_accuracy': float(overall_accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"测试集评估结果")
    print(f"{'=' * 60}")
    print(f"Overall Accuracy : {overall_accuracy:.4f}")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print(f"所有指标/图像已保存至: {output_dir}\\")
    print(f"  - loss_curve.png         训练损失 & 准确率曲线")
    print(f"  - confusion_matrix.png   归一化混淆矩阵热力图")
    print(f"  - per_class_metrics.png  每类 Precision/Recall/F1")
    print(f"  - metrics_summary.json   指标汇总 JSON")