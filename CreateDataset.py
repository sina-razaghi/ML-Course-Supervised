from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
import csv


def get_new_dataset():
    features, target = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=4,
        cluster_std=2
        )

    print("\nFeature Matrix: ")
    print(pd.DataFrame(features).head())
    print(features)

    print("\nTarget Class: ")
    print(pd.DataFrame(target).head())
    print(target)

    data = [[features[i][0], features[i][1], int(target[i])] for i in range(0, len(target))]
    # print(data)

    plt.scatter(features[:, 0], features[:, 1], marker="o", c=target, s=25, edgecolor="k")
    plt.show()
    return data

def save_dataset_scv(data, filename="example1"):
    with open(f'{filename}.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        names = ["Feature_1", "Feature_2", "Target"]
        writer.writerow(names)
        writer.writerows(data)
        print("\nSaved CSV !\n")



while True:
    getData = input("\nif you want new dataset enter 'y' or not 'n': ")
    if getData == 'y':
        dataset = get_new_dataset()
        saveData = input("\nif you want save this dataset enter 'y' or not 'n': ")
        if saveData == 'y':
            save_dataset_scv(dataset)
            break
        elif saveData == 'n':
            continue
    elif getData == 'n':
        break
