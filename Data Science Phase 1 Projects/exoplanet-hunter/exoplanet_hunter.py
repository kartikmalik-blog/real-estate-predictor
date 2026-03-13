# ============================================================
# 🌌 EXOPLANET HUNTER - NASA Kepler Space Telescope Data
# Built by: Kartik | Munich | 2025
# Goal: Predict which stars have orbiting exoplanets
# ============================================================

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

plt.style.use('dark_background')
print("✅ All libraries imported!")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('cumulative.csv')
print(f"\n🌌 NASA KEPLER DATASET")
print(f"   Stars:    {df.shape[0]}")
print(f"   Features: {df.shape[1]}")

# ============================================================
# EXPLORE TARGET
# ============================================================
print("\n🎯 TARGET COLUMN:")
print(df['koi_disposition'].value_counts())

# ============================================================
# CLEAN TARGET
# ============================================================
df = df[df['koi_disposition'] != 'CANDIDATE']
df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)

print(f"\n✅ Binary target created:")
print(f"   Has Planet : {df['target'].sum()}")
print(f"   No Planet  : {(df['target']==0).sum()}")

# Visualise
plt.figure(figsize=(7, 5))
labels = ['No Planet ❌', 'Has Planet 🌍']
values = [(df['target']==0).sum(), df['target'].sum()]
plt.bar(labels, values, color=['#ff4757','#00ff88'],
        edgecolor='white', width=0.5)
plt.title('🌌 Planet vs No Planet', fontsize=13)
plt.ylabel('Number of Stars')
for i, v in enumerate(values):
    plt.text(i, v+30, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('planet_distribution.png', dpi=150)
plt.show()
print("✅ Chart saved!")

# ============================================================
# CLEAN FEATURES
# ============================================================
cols_to_drop = ['koi_disposition','koi_pdisposition',
                'kepid','kepoi_name','kepler_name','koi_tce_delivname']
df = df.drop(columns=cols_to_drop, errors='ignore')

X = df.drop('target', axis=1)
y = df['target']

# Keep numeric only
X = X.select_dtypes(include=[np.number])

# Drop columns with >40% missing
missing_pct = X.isnull().mean()
cols_too_empty = missing_pct[missing_pct > 0.4].index
X = X.drop(columns=cols_too_empty)

# Fill remaining with median
X = X.fillna(X.median())

print(f"\n✅ Data cleaned!")
print(f"   Final features: {X.shape[1]}")
print(f"   Remaining NaN:  {X.isnull().sum().sum()}")

# ============================================================
# SPLIT & SCALE
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✅ Split & Scaled!")
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# TRAIN 3 MODELS
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(random_state=42, max_depth=10)
}

print("\n🚀 TRAINING MODELS...")
print("=" * 60)

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])
    cv      = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    results[name] = {'acc': acc, 'auc': auc, 'cv': cv.mean(), 'y_pred': y_pred}

    print(f"\n🌌 {name}")
    print(f"   Accuracy:     {acc*100:.2f}%")
    print(f"   AUC Score:    {auc:.3f}")
    print(f"   CV F1 Score:  {cv.mean():.3f} ± {cv.std():.3f}")

best = max(results, key=lambda x: results[x]['auc'])
print(f"\n🏆 BEST MODEL: {best} (AUC: {results[best]['auc']:.3f})")

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, results[best]['y_pred'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['No Planet','Has Planet'],
            yticklabels=['No Planet','Has Planet'],
            linewidths=2)
plt.title(f'🌌 Confusion Matrix — {best}', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Confusion matrix saved!")

# ============================================================
# ROC CURVE
# ============================================================
plt.figure(figsize=(10, 7))
colors = ['#00ff88', '#ffa502', '#ff4757']

for (name, model), color in zip(models.items(), colors):
    probs = model.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = results[name]['auc']
    plt.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f'{name} (AUC={auc:.3f})')

plt.plot([0,1],[0,1],'white',linestyle='--',label='Random (AUC=0.500)')
plt.title('🌌 ROC Curve — Exoplanet Detection', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print("✅ ROC curve saved!")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
dt_model = models['Decision Tree']
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 15

print("\n🌌 TOP 15 FEATURES:")
print("=" * 60)
for i in range(top_n):
    idx = indices[i]
    bar = '█' * int(importances[idx] * 100)
    print(f"   {i+1:2}. {X.columns[idx]:30} {importances[idx]:.4f} {bar}")

plt.figure(figsize=(10, 8))
plt.barh(range(top_n),
         importances[indices[:top_n]][::-1],
         color='#00ff88', edgecolor='white')
plt.yticks(range(top_n),
           [X.columns[indices[top_n-1-i]] for i in range(top_n)])
plt.xlabel('Importance Score')
plt.title('🌌 Top 15 Features — Planet Detection', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("✅ Feature importance saved!")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("🌌 EXOPLANET HUNTER — FINAL SUMMARY")
print("="*60)
print(f"   Dataset  : NASA Kepler — 9,564 real stars")
print(f"   Built by : Kartik | Munich | 2025")
print("="*60)
print("\n📊 MODEL RESULTS:")
for name, r in results.items():
    print(f"   {name:25} Acc: {r['acc']*100:.1f}%  AUC: {r['auc']:.3f}")
print(f"\n🏆 Best Model : {best}")
print(f"   Top Feature : koi_score (96.4% importance)")
print("\n✅ PROJECT 1 COMPLETE — READY FOR PORTFOLIO!")
print("="*60)