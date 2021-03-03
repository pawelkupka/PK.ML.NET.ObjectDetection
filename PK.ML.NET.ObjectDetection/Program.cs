using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace PK.ML.NET.ObjectDetection
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var images = ImageInputData.ReadFromFile("Data\\input");
            var imageDataView = mlContext.Data.LoadFromEnumerable(images);
            var modelScorer = new OnnxModelScorer("TinyYolo2_model.onnx", mlContext);
            var probabilities = modelScorer.Score(imageDataView);

            var parser = new OutputParser();
            var boundingBoxes = probabilities
                .Select(probability => parser.ParseOutputs(probability))
                .Select(boxes => parser.FilterBoundingBoxes(boxes, 10, .01f));

            DrawBoundingBoxes(images, boundingBoxes);
        }

        private static void DrawBoundingBoxes(IEnumerable<ImageInputData> images, IEnumerable<IList<BoundingBox>> boundingBoxes)
        {
            for (var i = 0; i < images.Count(); i++)
            {
                string imageFileName = images.ElementAt(i).Label;
                var detectedObjects = boundingBoxes.ElementAt(i);
                DrawBoundingBox("data\\input", "data\\output", imageFileName, detectedObjects);
            }
        }

        private static void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<BoundingBox> filteredBoundingBoxes)
        {
            var image = Image.FromFile(Path.Combine(inputImageLocation, imageName));
            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;
            foreach (var box in filteredBoundingBoxes)
            {
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;
                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";
                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);
                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);
                    thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                }
            }
            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }
            image.Save(Path.Combine(outputImageLocation, imageName));
        }
    }
}
