import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx


def collate_fn(batch):
    batch = batch[0].to(DEVICE)
    return batch


if __name__ == "__main__":

    ONNX_MODEL_PATH = "models/a2c_cartpole/actor.onnx"
    ESPDL_MODEL_PATH = "models/a2c_cartpole/actor.espdl"
    INPUT_SHAPE = [1, 4]
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    DEVICE = "cpu"

    x = torch.rand(1, 4)
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
