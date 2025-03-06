namespace Compression.tests.MGroup.LinearAlgebra.Tests.Utilities
{
    internal static class ArrayUtilities
    {
        internal static bool AreEqual(int[] expected, int[] computed)
        {
            if (expected.Length != computed.Length)
            {
                return false;
            }
            for (int i = 0; i < expected.Length; ++i)
            {
                if (expected[i] != computed[i])
                {
                    return false;
                }
            }
            return true;
        }

        internal static bool AreEqual(int[,] expected, int[,] computed)
        {
            if (expected.GetLength(0) != computed.GetLength(0) || expected.GetLength(1) != computed.GetLength(1))
            {
                return false;
            }
            for (int i = 0; i < expected.GetLength(0); ++i)
            {
                for (int j = 0; j < expected.GetLength(1); ++j)
                {
                    if (expected[i, j] != computed[i, j])
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        internal static double[,] GetSubmatrix(double[,] original, int[] rows, int[] columns)
        {
            int m = rows.Length;
            int n = columns.Length;
            var submatrix = new double[m, n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    submatrix[i, j] = original[rows[i], columns[j]];
                }
            }

            return submatrix;
        }

        internal static int[] ScaleAndRound(double[] original, double scaleFactor)
        {
            int n = original.Length;
            var result = new int[n];
            for (int i = 0; i < n; ++i)
            {
                result[i] = (int)Math.Round(original[i] * scaleFactor);
            }

            return result;
        }

        internal static int[,] ScaleAndRound(double[,] original, double scaleFactor = 1.0)
        {
            int m = original.GetLength(0);
            int n = original.GetLength(1);
            var result = new int[m, n];
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    result[i, j] = (int)Math.Round(original[i, j] * scaleFactor);
                }
            }

            return result;
        }

        /// <summary>
        /// Shallow copy.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="original"></param>
        /// <returns></returns>
        internal static T[] Copy<T>(T[] original)
        {
            var clone = new T[original.Length];
            Array.Copy(original, clone, original.Length);
            return clone;
        }

        /// <summary>
        /// Shallow copy.
        /// </summary>
        internal static T[,] Copy<T>(T[,] original)
        {
            var clone = new T[original.GetLength(0), original.GetLength(1)];
            Array.Copy(original, clone, original.Length);
            return clone;
        }

        internal static void SetAll<T>(T[,] array, T value)
        {
            for (int i = 0; i < array.GetLength(0); ++i)
            {
                for (int j = 0; j < array.GetLength(1); ++j)
                {
                    array[i, j] = value;
                }
            }
        }
    }
}
