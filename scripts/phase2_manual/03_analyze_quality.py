#!/usr/bin/env python3
"""
Phân tích chi tiết chất lượng dịch thuật dựa trên LaBSE embeddings
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Get project root (2 levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

print("\n" + "="*80)
print("PHÂN TÍCH CHI TIẾT CHẤT LƯỢNG DỊCH THUẬT - LaBSE")
print("="*80)
print()

# Load similarity analysis
print("Loading similarity analysis results...")
with open(PROJECT_ROOT / 'results/quality_analysis/similarity_analysis.json', 'r', encoding='utf-8') as f:
    analysis = json.load(f)

similarities = np.array(analysis['all_similarities'])
print(f"✓ Loaded {len(similarities)} samples\n")

# Load original data for context
print("Loading original data...")
with open(PROJECT_ROOT / 'data/manual_translations/vispider_train_2000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"✓ Loaded {len(data)} samples\n")

print("="*80)
print("1. PHÂN PHỐI SIMILARITY SCORES")
print("="*80)
print()

# Statistics
stats = analysis['overall_stats']
print(f"📊 Thống kê tổng thể:")
print(f"   Mean:              {stats['mean']:.4f}")
print(f"   Median:            {stats['percentile_50']:.4f}")
print(f"   Std deviation:     {stats['std']:.4f}")
print(f"   Min:               {stats['min']:.4f}")
print(f"   Max:               {stats['max']:.4f}")
print()

print(f"📈 Phân vị:")
print(f"   10th percentile:   {stats['percentile_10']:.4f}")
print(f"   25th percentile:   {stats['percentile_25']:.4f}")
print(f"   50th percentile:   {stats['percentile_50']:.4f}")
print(f"   75th percentile:   {stats['percentile_75']:.4f}")
print(f"   90th percentile:   {stats['percentile_90']:.4f}")
print()

# Distribution buckets
print("📊 Phân phối theo khoảng:")
buckets = {
    '0.90-1.00 (Excellent)': (0.90, 1.00),
    '0.80-0.90 (Very Good)': (0.80, 0.90),
    '0.70-0.80 (Good)': (0.70, 0.80),
    '0.60-0.70 (Acceptable)': (0.60, 0.70),
    '0.50-0.60 (Questionable)': (0.50, 0.60),
    '< 0.50 (Poor)': (0.0, 0.50)
}

for label, (low, high) in buckets.items():
    count = sum(1 for s in similarities if low <= s < high)
    pct = count / len(similarities) * 100
    print(f"   {label:<30s}: {count:4d} ({pct:5.1f}%)")

print()

print("="*80)
print("2. ĐÁNH GIÁ CHẤT LƯỢNG OVERALL")
print("="*80)
print()

# Quality assessment
excellent = sum(1 for s in similarities if s >= 0.85)
good = sum(1 for s in similarities if 0.75 <= s < 0.85)
acceptable = sum(1 for s in similarities if 0.65 <= s < 0.75)
questionable = sum(1 for s in similarities if s < 0.65)

print(f"✅ Excellent (≥0.85):    {excellent:4d} ({excellent/len(similarities)*100:5.1f}%)")
print(f"✓  Good (0.75-0.85):     {good:4d} ({good/len(similarities)*100:5.1f}%)")
print(f"⚠  Acceptable (0.65-0.75): {acceptable:4d} ({acceptable/len(similarities)*100:5.1f}%)")
print(f"❌ Questionable (<0.65): {questionable:4d} ({questionable/len(similarities)*100:5.1f}%)")
print()

# Overall quality verdict
total_good = excellent + good
if stats['mean'] >= 0.80:
    verdict = "✅ CHẤT LƯỢNG RẤT TỐT"
    detail = "Mean ≥ 0.80 cho thấy phần lớn bản dịch chính xác và giữ ngữ nghĩa tốt"
elif stats['mean'] >= 0.75:
    verdict = "✓ CHẤT LƯỢNG TỐT"
    detail = "Mean ≥ 0.75 là acceptable cho cross-lingual similarity"
elif stats['mean'] >= 0.70:
    verdict = "⚠ CHẤT LƯỢNG CHẤP NHẬN ĐƯỢC"
    detail = "Mean 0.70-0.75 có thể cần review thêm một số samples"
else:
    verdict = "❌ CẦN CẢI THIỆN"
    detail = "Mean < 0.70 cho thấy có vấn đề với quality hoặc embedding model"

print(f"ĐÁNH GIÁ TỔNG QUAN: {verdict}")
print(f"  {detail}")
print()

if total_good / len(similarities) >= 0.70:
    print(f"✓ {total_good/len(similarities)*100:.1f}% samples có quality tốt (≥0.75)")
    print("  → Dataset đáng tin cậy cho training/evaluation")
else:
    print(f"⚠ Chỉ {total_good/len(similarities)*100:.1f}% samples có quality tốt")
    print("  → Nên review và filter trước khi sử dụng")

print()

print("="*80)
print("3. PHÂN TÍCH THEO DIFFICULTY")
print("="*80)
print()

diff_stats = analysis['difficulty_stats']
print(f"{'Difficulty':<15} {'Count':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-"*80)

for diff in ['easy', 'medium', 'hard', 'extra_hard']:
    if diff in diff_stats:
        d = diff_stats[diff]
        print(f"{diff:<15} {d['count']:<8} {d['mean']:.4f}     {d['std']:.4f}     {d['min']:.4f}     {d['max']:.4f}")

print()

# Difficulty trend analysis
if all(d in diff_stats for d in ['easy', 'medium', 'hard', 'extra_hard']):
    means = [diff_stats[d]['mean'] for d in ['easy', 'medium', 'hard', 'extra_hard']]
    
    # Check if there's a decreasing trend
    if means[0] > means[-1] + 0.03:
        print("📉 Trend: Quality giảm dần khi difficulty tăng")
        print("   → Bình thường, câu khó dịch khó hơn")
    elif means[-1] > means[0] + 0.03:
        print("📈 Trend: Quality cao hơn ở câu khó!")
        print("   → Có thể do translator chú ý hơn với câu khó")
    else:
        print("➡️  Trend: Quality ổn định qua các mức độ khó")
        print("   → Rất tốt! Cho thấy consistency trong translation")

print()

print("="*80)
print("4. PHÂN TÍCH SAMPLES CÓ VẤN ĐỀ")
print("="*80)
print()

# Look at bottom samples
bottom_100 = analysis['bottom_100']
print(f"📋 {len(bottom_100)} samples có similarity thấp nhất:\n")

# Analyze why they have low similarity
print("Các lý do có thể:")

very_low = [s for s in bottom_100 if s['similarity'] < 0.50]
if very_low:
    print(f"\n⚠️  {len(very_low)} samples có similarity < 0.50:")
    print("   - Có thể dịch sai hoặc thiếu ngữ nghĩa")
    print("   - Nên kiểm tra thủ công")
    
    # Show examples
    print("\n   Examples:")
    for i, s in enumerate(very_low[:3], 1):
        print(f"\n   {i}. [{s['id']}] Similarity: {s['similarity']:.4f}")
        print(f"      EN: {s['question'][:70]}...")
        print(f"      VI: {s['vi_question'][:70]}...")

low_moderate = [s for s in bottom_100 if 0.50 <= s['similarity'] < 0.65]
if low_moderate:
    print(f"\n⚠️  {len(low_moderate)} samples có similarity 0.50-0.65:")
    print("   - Có thể do cấu trúc câu khác nhau")
    print("   - Hoặc dùng từ đồng nghĩa")
    print("   - Review để chắc chắn")

# Check for patterns in low similarity samples
print(f"\n📊 Phân bố theo difficulty trong bottom 100:")
diff_counter = Counter(s['hardness'] for s in bottom_100)
for diff in ['easy', 'medium', 'hard', 'extra_hard']:
    if diff in diff_counter:
        count = diff_counter[diff]
        pct = count / len(bottom_100) * 100
        expected_pct = diff_stats[diff]['count'] / len(data) * 100
        print(f"   {diff:<15}: {count:3d} ({pct:5.1f}%) - expected: {expected_pct:5.1f}%")
        if pct > expected_pct * 1.5:
            print(f"      ⚠️  Over-represented! Có thể {diff} samples khó dịch hơn")

print()

print("="*80)
print("5. KẾT LUẬN & KHUYẾN NGHỊ")
print("="*80)
print()

# Final recommendations
recommendations = []

if stats['mean'] >= 0.75:
    recommendations.append("✅ Mean similarity 0.75+ cho thấy quality tốt")
    recommendations.append("✅ Dataset sẵn sàng sử dụng cho training")
else:
    recommendations.append("⚠️  Mean similarity < 0.75 - nên xem xét")

if questionable > len(similarities) * 0.15:
    recommendations.append(f"⚠️  {questionable} samples (<0.65) nên được review thủ công")
    recommendations.append("   → Tạo filtered version cho production")
else:
    recommendations.append(f"✓ Chỉ {questionable} samples có vấn đề (<10%)")

if stats['std'] < 0.12:
    recommendations.append(f"✓ Std deviation {stats['std']:.4f} cho thấy consistency tốt")
else:
    recommendations.append(f"⚠️  Std deviation {stats['std']:.4f} hơi cao - có variance")

print("KHUYẾN NGHỊ:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print()

# Suggested thresholds
print("📏 NGƯỠNG KHUYẾN NGHỊ:")
print()
print(f"   High Quality (training):   similarity ≥ {stats['percentile_75']:.4f} ({sum(s >= stats['percentile_75'] for s in similarities)} samples)")
print(f"   Acceptable (validation):   similarity ≥ {stats['percentile_50']:.4f} ({sum(s >= stats['percentile_50'] for s in similarities)} samples)")
print(f"   Review Required:           similarity < 0.65 ({questionable} samples)")

print()

# Create distribution plot
print("="*80)
print("Creating visualization...")
print("="*80)
print()

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.4f}')
plt.axvline(stats['percentile_50'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["percentile_50"]:.4f}')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of LaBSE Similarity Scores')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot by difficulty
plt.subplot(1, 2, 2)
diff_data = []
diff_labels = []
for diff in ['easy', 'medium', 'hard', 'extra_hard']:
    if diff in diff_stats:
        indices = [i for i, d in enumerate(data) if d['hardness'] == diff]
        diff_sims = [similarities[i] for i in indices]
        diff_data.append(diff_sims)
        diff_labels.append(diff.replace('_', '\n'))

plt.boxplot(diff_data, labels=diff_labels)
plt.ylabel('Similarity Score')
plt.title('Similarity by Difficulty Level')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_png = PROJECT_ROOT / 'results/quality_analysis/labse_quality_analysis.png'
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_png}")

print()
print("="*80)
print("✅ PHÂN TÍCH HOÀN TẤT")
print("="*80)
print()
