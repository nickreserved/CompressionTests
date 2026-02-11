using Compression.src.MGroup.LinearAlgebra.Iterative.Stationary;
using DotNumerics.LinearAlgebra.CSLapack;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Iterative.Stationary.CSR;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Triangulation;
using MGroup.LinearAlgebra.Vectors;
using System.Diagnostics;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public class GeometricMultigridSolver : IGeometricMultigridSolver
    {
        public enum MatrixType { CSR, DUVI }

        private readonly IGeometricMultigridModel Model;
        public readonly bool GaussSeidel;
        private bool[] LevelDown;
        private int[] LevelIterations;
        private int firstLevel;
        private int totalLevels;
        private MatrixType matType;

        private Vector b;

        private IMatrixView[] LevelStiffness;
        private LdlSkyline coarseStiffnessLdlFactorized;
        private IMatrixView[] restriction;
        private IMatrixView[] interpolation;

        private Vector[] RelJacobiPreconditioner;

        private const int MaxCircleIterations = 10000;

        public int MaxIterations { get; set; }
        public double ConvergenceTolerance { get; set; }

        public int LevelSteps { get => LevelDown.Length; }

        public bool IsStepDown(int i) => LevelDown[i];

        public MatrixType MatType => matType;
        public int TotalLevels => totalLevels;
        public int NumDofsFree(int level) => LevelStiffness[level].NumRows;

        public static GeometricMultigridSolver CreateSimpleV(IGeometricMultigridModel model, bool GaussSeidel = true, MatrixType matType = MatrixType.CSR,
                                    int maxCircleIterations = MaxCircleIterations, bool coarseRelaxation = true, double convergenceTolerance = 1e-6, int fineLevelIterations = 4)
        {
            GeometricMultigridSolver a = new(model, GaussSeidel);
            a.Initialize(matType, maxCircleIterations, convergenceTolerance, new bool[] { true, false }, new int[] { fineLevelIterations }, coarseRelaxation);
            return a;
        }

        public static GeometricMultigridSolver CreateDeepV(IGeometricMultigridModel model, bool GaussSeidel = true, MatrixType matType = MatrixType.CSR,
                           int maxCircleIterations = MaxCircleIterations, bool coarseRelaxation = true, double convergenceTolerance = 1e-6, int depth = 2, int levelIterations = 4)
        {
            GeometricMultigridSolver a = new(model, GaussSeidel);
            a.Initialize(matType, maxCircleIterations,
                        convergenceTolerance,
                        Enumerable.Repeat(true, depth).Concat(Enumerable.Repeat(false, depth)).ToArray(), new int[] { levelIterations }, coarseRelaxation);
            return a;
        }        

        public static GeometricMultigridSolver CreateDeepV(IGeometricMultigridModel model, int[] levelIterations, bool GaussSeidel = true, MatrixType matType = MatrixType.CSR,
                            int maxCircleIterations = MaxCircleIterations, bool coarseRelaxation = true, double convergenceTolerance = 1e-6)
        {
            GeometricMultigridSolver a = new(model, GaussSeidel);
            a.Initialize(matType, maxCircleIterations,
                        convergenceTolerance,
                        Enumerable.Repeat(true, levelIterations.Length / 2).Concat(Enumerable.Repeat(false, levelIterations.Length / 2)).ToArray(), levelIterations, coarseRelaxation);
            return a;
        }

        public GeometricMultigridSolver(IGeometricMultigridModel model, bool GS) { Model = model; GaussSeidel = GS; }


        // calculate with power method -- only for tests -- it is inefficient
        private double EigenValueUpperBound(DokRowMajor AA)
        {
            const double percent = 0.1;
            CsrMatrix A = AA.ConvertToCsr();
            Vector x = Vector.CreateWithValue(A.NumRows, 1 / Math.Sqrt(A.NumRows));
            double l = 1;
            for (; ; )
            {
                Vector y = A.Multiply(x);
                double l2 = y.Norm2();
                x = y.Scale(1 / l2);
                double ratio = l2 / l;
                l = l2;
                if (ratio > 1 - percent && ratio < 1 + percent)
                {
                    if (ratio > 1) { l *= ratio; break; }
                    break;
                }
            }
            return l;
        }

        /// <summary>
        /// Calculates the Jacobi preconditioner of a sparse matrix.
        /// </summary>
        /// <param name="rows">An array of dictionaries column index -> value, one dictionary for each row of matrix.</param>
        /// <returns>The Jacobi preconditioner which is the inverse of the main diagonal of the matrix.</returns>
        internal static Vector JacobiPreconditioner(Dictionary<int, double>[] rows)
        {
            Vector x = Vector.CreateZero(rows.Length);
            for (int i = 0; i < rows.Length; ++i)
                x[i] = 1 / rows[i][i];
            return x;
        }

        /// <summary>
        /// Returns a relatively close upper bound for the spectral radius of a sparse matrix.
        /// </summary>
        /// <remarks>The matrix is <c>D^-1 * A</c> where <c>D^1</c> is the Jacobi preconditioner of <c>A</c> (the inverse of the diagonal matrix
        /// with main diagonal equal to main diagonal of matrix <c>A</c>) and <c>A</c> is the given matrix as array of row dictionaries.
        /// The result is the maximum of the sums of absolute values of rows of matrix <c>D^-1 * A</c></remarks>
        /// <param name="rows">An array of dictionaries column index -> value, one dictionary for each row of matrix.</param>
        /// <returns>An approximation of the upper bound for the eigenvalues of the matrix <c>D^-1 * A</c>.</returns>
        internal static double EigenValueUpperBound(Dictionary<int, double>[] rows)
        {
            double l = 0;   // max eigenvalue
            for (int i = 0; i < rows.Length; ++i)
                l = Math.Max(l, rows[i].Values.Select(v => Math.Abs(v)).Sum() / Math.Abs(rows[i][i])); // approximation of max eigenvalue
            return l;
        }

        /// <summary>
        /// Under or over relaxes the Jacobi preconditioner of a matrix.
        /// </summary>
        /// <remarks>Actually it multiplies the jacobi preconditioner with scalar <c>w = 1 / lmax</c> where <c>lmax</c> is an approximate
        /// upper bound of the spectral radius of matrix D^-1 * A, where D^-1 = x.
        /// We can assume that <c>lmax</c> is larger as approximate upper bound.</remarks>
        /// <param name="x">The Jacobi preconditioner of a matrix, which is <c>D^-1</c> the inverse of the main diagonal of the matrix <c>A</c>.
        /// After the call, it is under or over relaxed.</param>
        /// <param name="l">An approximation of the upper bound for the eigenvalues of the matrix.</param>
        internal static void RelaxateJacobiPreconditioner(Vector x, double l)
        {
            l = 2 / l;
            for (int i = 0; i < x.Length; ++i)
                x[i] *= l;
        }
        
        /// <summary>
        /// Generates a procedure-friendly geometric multigrid path between level of detail.
        /// </summary>
        /// <param name="totalLevels">Returns the number of total levels of detail in generated geometric multigrid path.</param>
        /// <param name="LevelDown">An array of steps. <c>true</c> means go to coarser level of detail and <c>false</c> means go to finer
        /// level of detail. First element can be in different level of detail than finest. Last element can be in different level of detail
        /// than finest. On input, last element can be in different level of detail than the first element, but on output this is fixed with
        /// padding (if required) of some elements <c>true</c> or <c>false</c>.</param>
        /// <param name="firstLevel">Returns the number of the level of detail for the first element of <paramref name="LevelDown"/></param>
        internal static void MakePath(ref int totalLevels, ref bool[] LevelDown, ref int firstLevel)
        {
            // Make the path
            int currentLevel = 0;
            totalLevels = 1; firstLevel = 0;
            for (int i = 0; i < LevelDown.Length; ++i)
            {
                if (LevelDown[i])
                {
                    ++currentLevel;
                    if (currentLevel == totalLevels) ++totalLevels;
                }
                else if (currentLevel == 0)
                {
                    ++firstLevel;
                    ++totalLevels;
                }
                else --currentLevel;
            }
            if (currentLevel > firstLevel)
                LevelDown = LevelDown.Concat(Enumerable.Repeat(false, currentLevel - firstLevel)).ToArray();
            else if (currentLevel < firstLevel)
                LevelDown = LevelDown.Concat(Enumerable.Repeat(true, firstLevel - currentLevel)).ToArray();
        }

        /// <summary>
        /// Over- or under-relaxate the Jacobi preconditioner.
        /// </summary>
        /// <param name="A">The matrix as array of row dictionaries.</param>
        /// <return>The Jacobi preconditioner of matrix <paramref name="A"/>, over- or under-relaxed <c>w * D^-1</c>.</return>
        internal static Vector RelaxedJacobiPreconditioner(Dictionary<int, double>[] A)
        {
            Vector v = JacobiPreconditioner(A);
            double l = EigenValueUpperBound(A); // calculate upper bound of eigenvalues
            RelaxateJacobiPreconditioner(v, l);
            return v;
        }
        /// <summary>
        /// Over- or under-relaxate the Jacobi preconditioners on each level of detail.
        /// </summary>
        /// <param name="A">The matrix as array of row dictionaries for the coarser level of detail.</param>
        /// <param name="RelJacobiPreconditioner">An array of the Jacobi preconditioners of all levels of detail (except the most coarse),
        /// as input. The over- or under-relaxed Jacobi preconditioners <c>w * D^-1</c> for output.</param>
        internal static void RelaxateJacobiPreconditioners(DokRowMajor A, Vector[] RelJacobiPreconditioner)
        {
            double l = EigenValueUpperBound(A.RawRows); // calculate upper bound of eigenvalues in smaller matrix
            foreach (var v in RelJacobiPreconditioner)
                RelaxateJacobiPreconditioner(v, l);
        }

        public void Initialize(MatrixType matType, int maxCircleIterations, double convergenceTolerance, bool[] levelDown, int[] levelIterations, bool coarseRelaxation)
        {
            this.matType = matType;
            LevelDown = levelDown;
            LevelIterations = levelIterations;
            MaxIterations = maxCircleIterations;
            ConvergenceTolerance = convergenceTolerance;

            MakePath(ref totalLevels, ref LevelDown, ref firstLevel);

            // Generate coarser models
            LevelStiffness = new IMatrixView[totalLevels - 1];
            restriction = new IMatrixView[totalLevels - 1];
            interpolation = new IMatrixView[totalLevels - 1];
            if (!GaussSeidel) RelJacobiPreconditioner = new Vector[totalLevels - 1];
            (DokRowMajor A, b) = Model.CreateLinearSystem();
            IGeometricMultigridModel currentModel = Model;
            for (int i = 0; i < totalLevels - 1; ++i)
            {
                //if (!GaussSeidel) HealVector[i] = HealStiffnessMatrix(A.RawRows);
                if (!GaussSeidel) RelJacobiPreconditioner[i] = coarseRelaxation ? JacobiPreconditioner(A.RawRows)
                                                                                : RelaxedJacobiPreconditioner(A.RawRows);
                (currentModel, DokRowMajor restrictionB, DokRowMajor interpolationB) = currentModel.CreateCoarserModelAndSmoothenerMatrices();
                switch(matType)
                {
                    case MatrixType.CSR:
                        LevelStiffness[i] = A.BuildCsrMatrix(true);
                        restriction[i] = restrictionB.BuildCsrMatrix(true);
                        interpolation[i] = interpolationB.BuildCsrMatrix(true);
                        break;
                    case MatrixType.DUVI:
                        LevelStiffness[i] = new DuViCompressedSparseMatrix(A);
                        restriction[i] = new DuViCompressedSparseMatrix(restrictionB);
                        interpolation[i] = new DuViCompressedSparseMatrix(interpolationB);
                        break;
                }
                (A, _) = currentModel.CreateLinearSystem();
            }
            if (!GaussSeidel && coarseRelaxation) RelaxateJacobiPreconditioners(A, RelJacobiPreconditioner);
            coarseStiffnessLdlFactorized = SkylineMatrix.CreateFromMatrix(A.BuildCsrMatrix(true).CopyToFullMatrix(), 1e-15).FactorLdl(true, 1e-15);

#if DEBUG
            //OutputMatrix(LevelStiffness);
            File.Delete(GetLogPath());
#endif
        }

        private static void OutputMatrix(IMatrixView[] A)
        {
#if DEBUG
            string path = "csr_matrices_cpu.txt";
            File.Delete(path);
            for (int i = 0; i < A.Length; ++i)
            {
                if (i == 1)
                {
                    Matrix B = A[i].CopyToFullMatrix();
                    string line = "stiffness_matrix(" + i + ") = [" + Environment.NewLine;
                    for (int m = 0; m < B.NumRows; ++m)
                        line += string.Join(" ", B.GetRow(m).RawData.Select(e => e.ToString(/*"G4"*/))) + Environment.NewLine;
                    line += "]" + Environment.NewLine + Environment.NewLine;
                    //                    line += "_matrix(" + k + ") = [" + string.Join(Environment.NewLine, A[i].CopyToFullMatrix().RawData.Select(e => e.ToString("G6"))) + "]" + Environment.NewLine;
                    File.AppendAllText(path, line);
                }
            }
#endif
        }

        private string GetLogPath() => GetLogPath(matType == MatrixType.DUVI, GaussSeidel, false);
        internal static string GetLogPath(bool duvi, bool gaussSeidel, bool gpu)
            => "output_" + (duvi ? "duvi" : "csr") + "_" + (gaussSeidel ? "gauss_seidel" : "jacobi") + "_" + (gpu ? "g" : "c") + "pu.txt";

        private void OutputVectorX(int currentLevel, Vector[] x, string name) => OutputVectorX(currentLevel, x[currentLevel], name);
        private void OutputVectorX(int currentLevel, Vector x, string name)
        {
#if DEBUG
            string line = name + "(" + currentLevel + ") = [" + string.Join(" ", x.RawData.Select(e => e.ToString("G14"))) + "]" + Environment.NewLine;
            File.AppendAllText(GetLogPath(), line);
#endif
        }

        public (Vector, IterativeStatistics, double[]) Solve(Vector? xInitialGuess)
        {
            IStationaryIteration[] stationaryIteration = null;
            if (GaussSeidel)
            {
                stationaryIteration = new IStationaryIteration[totalLevels - 1];
                for (int i = 0; i < stationaryIteration.Length; ++i)
                {
                    stationaryIteration[i] = matType == MatrixType.CSR ? new GaussSeidelIterationCsr() : new GaussSeidelIterationCsrDuVi();
                    stationaryIteration[i].UpdateMatrix(LevelStiffness[i], false);
                }
            }

            Vector[] x = new Vector[totalLevels - 1];
            if (xInitialGuess == null) xInitialGuess = Vector.CreateZero(NumDofsFree(0));
            x[0] = xInitialGuess;

            Vector[] r = new Vector[totalLevels];
            r[0] = b;
            // if first level is not 0
            for (int i = 0; i < firstLevel; ++i)
                r[i + 1] = (Vector) restriction[i].Multiply(r[i]);

            double[] time = new double[totalLevels];
            Stopwatch stopwatch = new();

            // loop of cycles
            for (int iterations = 0; ; ++iterations)
            {
                int currentLevel = firstLevel;

                // loop of steps inside a cycle
                for (int step = 0; step < LevelSteps; ++step)
                {
                    // Multigrid level time count
                    stopwatch.Restart();

                    if (currentLevel == totalLevels - 1)
                    {
                        OutputVectorX(currentLevel, r, "B");
                        Vector eCoarse = coarseStiffnessLdlFactorized.SolveLinearSystem(r[currentLevel]);
                        OutputVectorX(currentLevel, eCoarse, "X");
                        Vector eFine = (Vector) interpolation.Last().Multiply(eCoarse);
                        x[currentLevel - 1].AddIntoThis(eFine);
                        OutputVectorX(currentLevel - 1, x, "X");
                    }
                    else
                    {
                        // try to solve A * xInitialGuess = b
                        int lvlIter = LevelIterations[Math.Min(currentLevel, LevelIterations.Length - 1)];
                        for (int i = 0; i < lvlIter; ++i)
                        {
                            if (GaussSeidel)
                                stationaryIteration[currentLevel].Execute(r[currentLevel], x[currentLevel]);
                            else
                            {
                                // x += w * D^-1 * (b - A * x)
                                IVector res = Vector.CreateZero(x[currentLevel].Length);
                                LevelStiffness[currentLevel].MultiplyIntoResult(x[currentLevel], res);
                                res.SubtractIntoThis(r[currentLevel]);
                                for (int j = 0; j < res.Length; ++j)
                                    x[currentLevel][j] -= res[j] * RelJacobiPreconditioner[currentLevel][j];
                            }
                        }
                        if (currentLevel == 1 && !LevelDown[step]) OutputVectorX(currentLevel, x, "x");

                        if (LevelDown[step])
                        {
                            // calculate residual
                            Vector rFine = r[currentLevel].Copy();
                            rFine.SubtractIntoThis(LevelStiffness[currentLevel].Multiply(x[currentLevel]));

                            // small residual or exceeded the iteration number
                            if (currentLevel == 0)
                            {
                                bool converged = true;
                                bool failed = false;
                                for (int i = 0; i < rFine.Length; ++i)
                                {
                                    double p = Math.Abs(rFine[i]);
                                    if (Double.IsNaN(p) || p > 1e50) { failed = true; converged = false; break; }  
                                    else if (p > ConvergenceTolerance) { converged = false; break; } 
                                }
                                double residual = rFine.Norm2();
                                if (converged || failed || iterations > MaxIterations)
                                {
                                    IterativeStatistics stats = new()
                                    {
                                        NumIterationsRequired = iterations,
                                        ConvergenceCriterion = ("dumb text", residual),
                                        HasConverged = converged
                                    };
                                    return (x[0], stats, time);
                                }
                            }

                            OutputVectorX(currentLevel, r, "B");
                            OutputVectorX(currentLevel, x, "X");
                            OutputVectorX(currentLevel, rFine, "R");

                            // fine residual to coarse residual
                            r[currentLevel + 1] = (Vector) restriction[currentLevel].Multiply(rFine);
                            if (currentLevel < totalLevels - 2)
                                x[currentLevel + 1] = Vector.CreateZero(r[currentLevel + 1].Length);
                        }
                        else
                        {
                            Vector eFine = (Vector) interpolation[currentLevel - 1].Multiply(x[currentLevel]);
                            x[currentLevel - 1].AddIntoThis(eFine);

                            OutputVectorX(currentLevel, x, "X");
                            OutputVectorX(currentLevel - 1, x, "X");
                        }
                    }
                    // Multigrid level time count
                    stopwatch.Stop();
                    time[currentLevel] += stopwatch.Elapsed.TotalMilliseconds;

                    currentLevel += LevelDown[step] ? 1 : -1;
                }
            }
        }

    }
}
