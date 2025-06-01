from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "./Llama-3.2-3B-Instruct-p20-ft"
quant_path = "Llama-3.2-3B-Instruct-gptq-b4g32-p20"

calibration_dataset = load_dataset(
    "Salesforce/wikitext",
    data_files="wikitext-2-raw-v1/train-00000-of-00001.parquet",
    split="train"
  ).select(range(2048))["text"]

quant_config = QuantizeConfig(bits=4, group_size=32, v2=True)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)