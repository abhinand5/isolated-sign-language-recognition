import timeit


def predict_on_batch(model, X, non_empty_frame_idxs):
    model.predict_on_batch(
        {"frames": X[:1], "non_empty_frame_idxs": non_empty_frame_idxs[:1]}
    )

def benchmark_train(model, X, non_empty_frame_idxs):
    print(timeit.timeit(predict_on_batch, number=100) / 100)
