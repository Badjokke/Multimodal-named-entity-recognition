# Configuration file
The configuration is **case sensitive**. The configuration provided runs everything and points to directories where they should be by default. If you move them around, change the paths here.
```json
{
  "experiments": [ --- Array of experiment objects which will be run sequentially
    {
      "pipeline": [ --- Which pipeline should be run for each provided dataset
        "TEXT",
        "IMAGE",
        "MULTIMODAL"
      ],
      "datasets": { --- Which datasets should be used in experiments, keys are also case sensitive.
        "SOA": { ---  Custom dataset from books
          "input_path": "../dataset/preprocessed/SOA" --- Path to root directory of the dataset, the child directories must be text_preprocessed and image_preprocessed
        },
        "T15": {
          "input_path": "../dataset/preprocessed/twitter_2015"
        },
        "T17": {
          "input_path": "../dataset/preprocessed/twitter_2017"
        }
      },
      "models": [ --- all available models, for multimodal pipeline, BERT LLAMA and LSTM values are used to run both variations of multimodal models (i.e. the one with VIT aswell the one with CNN).
        "BERT",
        "LLAMA",
        "LSTM",
        "VIT", --- VIT and CNN refer to image only classifier, they do not impact the multimodal
        "CNN"
      ],
      "results_path": "../mner_experiments" -- root directories where results of the experiments (state dictionaries and figures) will be dumped
    }
  ]
}
```