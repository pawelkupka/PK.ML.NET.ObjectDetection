using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace PK.ML.NET.ObjectDetection
{
    public class ImageInputData
    {
        [LoadColumn(0)]
        public string Source;

        [LoadColumn(0)]
        public string Label;

        public static IEnumerable<ImageInputData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageInputData { Source = filePath, Label = Path.GetFileName(filePath) });
        }
    }
}
