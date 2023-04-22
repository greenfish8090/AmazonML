import numpy as np
from model import TransformerRegressor, TransformerEntityRegressor
import torch
import pandas as pd
from dataset import TextDataset, TextEEDataset
from tqdm import tqdm

def main():
    # model = TransformerRegressor("bert-base-uncased").to("cuda")
    # loaded = torch.load("checkpoints/e2e_t/model_best_iter_16000.pth.tar", map_location="cuda")
    # model.load_state_dict(loaded["state_dict"])
    # print("Loaded model")
    # model.eval()

    # test_set = TextDataset("dataset/test.csv", test=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
    # test_preds = pd.DataFrame({'PRODUCT_ID': test_set.data['PRODUCT_ID'], 'PRODUCT_LENGTH': 0})
    # total = 0
    # with torch.no_grad():
    #     for i, x in enumerate(tqdm(test_loader)):
    #         B = len(x["string"])
    #         out = model(x, "cuda")
    #         test_preds.loc[total:total+B-1, 'PRODUCT_LENGTH'] = out.detach().cpu().squeeze().numpy()
    #         total += B
    #         print(i, total, flush=True)

    #         if i % 500 == 0:
    #             test_preds.to_csv(f"dataset/e2e_preds_{i}.csv", index=False)
    # test_preds.to_csv("dataset/e2e_preds.csv", index=False)

    df = pd.read_csv("dataset/split_train.csv")
    vc = dict(df["PRODUCT_TYPE_ID"].value_counts())
    id_to_ind = {}
    default_ind = 0
    for k, v in vc.items():
        if v > 10:
            id_to_ind[k] = default_ind
            default_ind += 1
        else:
            id_to_ind[k] = default_ind
    test_set = TextEEDataset(
        path="dataset/test.csv",
        id_to_ind=id_to_ind,
        default_ind=default_ind,
        test=True,
        transform=True,
    )

    model = TransformerEntityRegressor("bert-base-uncased", 32, len(id_to_ind)).to("cuda")
    loaded = torch.load("checkpoints/e2e_t/iter_464000.pth.tar", map_location="cuda")
    model.load_state_dict(loaded["state_dict"])
    print("Loaded model")
    model.eval()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
    test_preds = pd.DataFrame({'PRODUCT_ID': test_set.data['PRODUCT_ID'], 'PRODUCT_LENGTH': 0})
    total = 0
    with torch.no_grad():
        for i, (string, x) in enumerate(tqdm(test_loader)):
            B = len(x)
            out = model(string, x, "cuda")
            out = out.detach().cpu().squeeze().numpy()
            test_preds.loc[total:total+B-1, 'PRODUCT_LENGTH'] = np.exp(out * test_set.std + test_set.mean)
            total += B
            print(i, total, flush=True)

            if i % 500 == 0:
                test_preds.to_csv(f"dataset/e2e_t_454k_preds_{i}.csv", index=False)
    test_preds.to_csv("dataset/e2e_t_464k_preds.csv", index=False)
if __name__ == "__main__":
    main()




    