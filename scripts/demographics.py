import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    hdata = pd.read_csv("OUTCOME_DIAGNOSIS_processed.csv", header=0)

    ages = hdata['AgeC']

    img_root = "imgs/demographics/"

    if (not os.path.exists(img_root)):
        os.makedirs(img_root)

    plt.hist(ages, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Age (years)')
    plt.ylabel('Number of Participants')
    plt.title('Age Distribution of HELIAD Participants')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{img_root}/age_distribution.png")

    plt.figure(figsize=(10, 6))
    age_counts = ages.value_counts().sort_index()
    age_counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Age (years)')
    plt.ylabel('Number of Participants')
    plt.title('Age Distribution of HELIAD Participants')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{img_root}/age_distribution_valcounts.png")

    genders = hdata['SexC']
    genders_text = genders.replace({1: 'Male',   2: 'Female'})
    
    male_count = (genders_text == 'Male').sum()
    female_count = (genders_text == 'Female').sum()
    total = male_count + female_count
    scale_factor = 35000
    male_size = male_count / total * scale_factor
    female_size = female_count / total * scale_factor
    min_size = 3000
    male_size = max(male_size, min_size)
    female_size = max(female_size, min_size)

    plt.figure(figsize=(8, 6))

    plt.axis('off')

    plt.scatter(0.35, 0.5, s=male_size, alpha=0.7, color='skyblue', 
                edgecolor='navy', linewidth=0, zorder=2)
    plt.scatter(0.65, 0.5, s=female_size, alpha=0.7, color='lightcoral', 
                edgecolor='darkred', linewidth=0, zorder=2)

    plt.text(0.35, 0.5, f'Male\n\n{male_count:,}', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='navy', zorder=3)
    plt.text(0.65, 0.5, f'Female\n\n{female_count:,}', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='darkred', zorder=3)

    plt.title(f'Heliad Study\nBiological Gender Distribution\nTotal Participants: {total:,}', 
            fontsize=16, fontweight='bold', pad=30)

    plt.xlim(0.1, 0.9)
    plt.ylim(0.2, 0.8)

    plt.tight_layout()
    plt.savefig(f"{img_root}/participant_gender_distribution.png")

    hdata['SexD'] = hdata['SexC'].replace({1: 'Male', 2: 'Female'})

    plt.figure(figsize=(12, 6))

    sns.kdeplot(data=hdata, x='AgeC', hue='SexD', 
            fill=True, alpha=0.3, linewidth=2)

    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    plt.title('Age Distribution Density by Gender - HELIAD Study')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{img_root}/participant_age_density_gender.png")
    
    hdata['age_group'] = pd.cut(hdata['AgeC'], 
                         bins=[60, 70, 80, 90, 100],
                         labels=['60-69', '70-79', '80-89', '90+'])

    plt.figure(figsize=(14, 7))

    # Create box plot
    sns.boxplot(x='age_group', y='BMI', hue='SexC', data=hdata, palette=['skyblue', 'lightcoral'])

    plt.xlabel('Age Group (years)')
    plt.ylabel('BMI (kg/m²)')
    plt.title('BMI Distribution by Age Group and Gender - HELIAD Study', fontsize=14, fontweight='bold')
    plt.legend(title='Gender')
    plt.grid(True, alpha=0.3, axis='y')

    age_gender_counts = hdata.groupby(['age_group', 'SexC']).size().unstack()
    for i, age_group in enumerate(hdata['age_group'].cat.categories):
        if age_group in age_gender_counts.index:
            male_count = age_gender_counts.loc[age_group, 'Male'] if 'Male' in age_gender_counts.columns else 0
            female_count = age_gender_counts.loc[age_group, 'Female'] if 'Female' in age_gender_counts.columns else 0
            plt.text(i-0.2, hdata['BMI'].min() - 2, f'n={male_count}', 
                    ha='center', va='top', fontsize=9, color='blue')
            plt.text(i+0.2, hdata['BMI'].min() - 2, f'n={female_count}', 
                    ha='center', va='top', fontsize=9, color='red')

    plt.ylim([hdata['BMI'].min() - 3, hdata['BMI'].max() + 2])
    plt.tight_layout()
    plt.savefig(f"{img_root}/bmi_per_gender_age.png")

if __name__ == "__main__":    
    main()