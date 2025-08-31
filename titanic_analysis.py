#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # للاستخدام في سطر الأوامر
import matplotlib.pyplot as plt
import seaborn as sns

# تحميل البيانات
print("جارٍ تحميل بيانات تيتانيك...")
df = pd.read_csv('titanic.csv')

# نظرة عامة على البيانات
print("\n=== نظرة عامة على البيانات ===")
print("الأبعاد:", df.shape)
print("\nأول 10 صفوف:")
print(df.head(10))
print("\nمعلومات البيانات:")
print(df.info())
print("\nالوصف الإحصائي:")
print(df.describe())

# القيم المفقودة
print("\n=== القيم المفقودة ===")
missing_values = df.isnull().sum()
print("القيم المفقودة في كل عمود:")
print(missing_values)

# تحليل الديموغرافيا
print("\n=== تحليل الديموغرافيا ===")
avg_age = df['Age'].mean()
print(f"متوسط العمر: {avg_age:.2f} سنة")

youngest = df['Age'].min()
oldest = df['Age'].max()
print(f"أصغر راكب: {youngest} سنة")
print(f"أكبر راكب: {oldest} سنة")

# توزيع الأعمار
plt.figure(figsize=(10, 6))
plt.hist(df['Age'].dropna(), bins=20, edgecolor='black', alpha=0.7)
plt.title('توزيع أعمار ركاب تيتانيك')
plt.xlabel('العمر')
plt.ylabel('التكرار')
plt.grid(True, alpha=0.3)
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# تحليل النجاة
print("\n=== تحليل النجاة ===")
survival_rate = df['Survived'].mean() * 100
print(f"معدل النجاة العام: {survival_rate:.2f}%")

# النجاة حسب الجنس
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
print("\nمعدل النجاة حسب الجنس:")
print(gender_survival)

plt.figure(figsize=(8, 6))
gender_survival.plot(kind='bar', color=['pink', 'blue'])
plt.title('معدل النجاة حسب الجنس')
plt.ylabel('نسبة النجاة (%)')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.savefig('gender_survival.png', dpi=300, bbox_inches='tight')
plt.close()

# النجاة حسب فئة الركاب
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print("\nمعدل النجاة حسب فئة الركاب:")
print(class_survival)

plt.figure(figsize=(8, 6))
class_survival.plot(kind='bar', color=['gold', 'silver', 'brown'])
plt.title('معدل النجاة حسب فئة الركاب')
plt.ylabel('نسبة النجاة (%)')
plt.xlabel('فئة الركاب')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.savefig('class_survival.png', dpi=300, bbox_inches='tight')
plt.close()

# النجاة حسب ميناء الصعود
embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100
print("\nمعدل النجاة حسب ميناء الصعود:")
print(embarked_survival)

# تحليل العائلة والأجرة
print("\n=== تحليل العائلة والأجرة ===")
df['Alone'] = (df['SibSp'] + df['Parch'] == 0).astype(int)
alone_survival = df.groupby('Alone')['Survived'].mean() * 100
print("\nمعدل النجاة حسب السفر بمفردك أو مع العائلة:")
print(alone_survival)

# متوسط الأجرة للناجين vs غير الناجين
fare_survival = df.groupby('Survived')['Fare'].mean()
print("\nمتوسط الأجرة للناجين وغير الناجين:")
print(fare_survival)

# توزيع الأجرة
plt.figure(figsize=(10, 6))
plt.hist(df['Fare'], bins=30, edgecolor='black', alpha=0.7)
plt.title('توزيع أجرة الركاب')
plt.xlabel('الأجرة')
plt.ylabel('التكرار')
plt.grid(True, alpha=0.3)
plt.savefig('fare_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# تحليل الفئات العمرية
print("\n=== تحليل الفئات العمرية ===")
bins = [0, 12, 19, 59, 100]
labels = ['أطفال (0-12)', 'مراهقون (13-19)', 'بالغون (20-59)', 'كبار السن (60+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

agegroup_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
print("\nمعدل النجاة حسب الفئة العمرية:")
print(agegroup_survival)

plt.figure(figsize=(10, 6))
agegroup_survival.plot(kind='bar', color=['lightblue', 'lightgreen', 'orange', 'pink'])
plt.title('معدل النجاة حسب الفئة العمرية')
plt.ylabel('نسبة النجاة (%)')
plt.xlabel('الفئة العمرية')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.savefig('agegroup_survival.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== الخلاصة ===")
print("من خلال تحليل بيانات تيتانيك، يمكننا استنتاج ما يلي:")
print("1. كان معدل النجاة العام حوالي 38%")
print("2. كانت نسبة نجاة الإناث أعلى بكثير من الذكور (حوالي 74% مقابل 19%)")
print("3. كان لركاب الدرجة الأولى أعلى معدل نجاة يليهم ركاب الدرجة الثانية ثم الثالثة")
print("4. الركاب الذين سافروا بمفردهم كان معدل نجاتهم أقل من الذين سافروا مع عائلاتهم")
print("5. كان متوسط أجرة الناجين أعلى من غير الناجين")
print("6. كان للأطفال والنساء الأولوية في عمليات الإنقاذ")

print("\nتم حفظ المخططات البيانية في ملفات PNG في المجلد الحالي")
