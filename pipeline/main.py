import select_model_and_dataset

if __name__ == "__main__":
    model_id, dataset_id = select_model_and_dataset.run()
    print(model_id, dataset_id)