import pandas as pd

def main():

    filepath = 'data/Eval/list_eval_partition.txt'
    all_images = pd.read_csv(filepath, sep=" ", header=None)
    all_images.columns = ['image', 'dataset']

    # Append subdirectory in path so that it works with the keras ImageGenerator
    all_images['image'] = all_images['image'].apply(lambda x: "img_align_celeba/{}".format(x))


    df_train = all_images.loc[all_images.dataset == 0]
    df_val = all_images.loc[all_images.dataset == 1]
    df_test = all_images.loc[all_images.dataset == 2]

    df_train.to_csv('data/Eval/train.csv', index=False)
    df_val.to_csv('data/Eval/val.csv', index=False)
    df_test.to_csv('data/Eval/test.csv', index=False)


if __name__ == "__main__":
    main()



