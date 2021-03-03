using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace PK.ML.NET.ObjectDetection
{
    public class OnnxModelScorer
    {
        private readonly string _modelLocation;
        private readonly MLContext _mlContext;

        public OnnxModelScorer(string modelLocation, MLContext mlContext)
        {
            _modelLocation = modelLocation;
            _mlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 416;
            public const int imageWidth = 416;
        }

        public struct ModelSettings
        {
            public const string ModelInput = "image";
            public const string ModelOutput = "grid";
        }

        private ITransformer LoadModel(string modelLocation)
        {
            var data = _mlContext.Data.LoadFromEnumerable(new List<ImageInputData>());
            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageInputData.Source))
                            .Append(_mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
                            .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
                            .Append(_mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { ModelSettings.ModelOutput }, inputColumnNames: new[] { ModelSettings.ModelInput }));

            var model = pipeline.Fit(data);
            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            var scoredData = model.Transform(testData);
            var probabilities = scoredData.GetColumn<float[]>(ModelSettings.ModelOutput);
            return probabilities;
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(_modelLocation);
            return PredictDataUsingModel(data, model);
        }
    }
}
