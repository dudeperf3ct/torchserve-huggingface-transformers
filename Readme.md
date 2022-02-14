# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

## Setup

TorchServe is avaliable as docker image `pytorch/torchserve`.

```bash
docker pull pytorch/torchserve
docker run --rm -it  pytorch/torchserve:latest bash
```

## Run

### Store a Model

To serve a model with TorchServe, first archive the model as a MAR file. You can use the model archiver to package a model. You can also create model stores to store your archived models.

- Get model and other files related to transformer models. `pytorch_model.bin`, `vocab.txt` and `config.json` file.

```bash
./get_model.sh
```

Now we will package the model inside torchserve docker container.

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples -v $(pwd)/transformer_handler.py:/home/model-server/transformer_handler.py  pytorch/torchserve:latest bash
```

Inside docker run `torch-model-archiver` to get `.mar` format model which will be used for inference. `torch-model-archiver` takes model checkpoints or model definition file with state_dict, and package them into a `.mar` file. This file can then be redistributed and served by anyone using TorchServe.

```bash
torch-model-archiver --model-name SentimentClassification --version 1.0 --serialized-file model-store/pytorch_model.bin --handler ./transformer_handler.py --extra-files "model-store/config.json,model-store/vocab.txt,examples/index_to_name.json"
```

### Register and serve the model

After you archive and store the model, to register the model on TorchServe using the above model archive file, we run the following commands:

```bash
mv SentimentClassification.mar model-store/
torchserve --start --model-store model-store --models my_tc=SentimentClassification.mar --ncs
```

### Get predictions from a model

In a separate terminal,

```bash
curl -X POST http://127.0.0.1:8080/predictions/my_tc -T examples/sample_text1.txt
curl -X POST http://127.0.0.1:8080/predictions/my_tc -T examples/sample_text0.txt 
```

### Stop TorchServe

To stop the currently running TorchServe instance, run

```bash
torchserve --stop
```

### Inspect the logs

All the logs you've seen as output to stdout related to model registration, management, inference are recorded in the `/logs` folder.

High level performance data like Throughput or Percentile Precision can be generated with [Benchmark](https://github.com/pytorch/serve/blob/master/benchmark/README.md) and visualized in a report.

Additionals:

**Batch Inference**: [Batching](https://github.com/pytorch/serve/tree/0c5daa3a486bb3763bed0954df5e7abc24162cac/examples/Huggingface_Transformers#batch-inference)

**Inference**: Supports inference through both [gRPC](https://github.com/pytorch/serve/blob/master/docs/grpc_api.md) and [HTTP/REST](https://github.com/pytorch/serve/blob/master/docs/rest_api.md).
