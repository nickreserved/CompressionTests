using CASS.OpenCL;
using Compression.src.MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Iterative.Stationary.CSR;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Triangulation;
using MGroup.LinearAlgebra.Vectors;
using System.Diagnostics;
using System.Xml.Linq;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public class GeometricMultigridSolver : IGeometricMultigridSolver
    {
        public enum MatrixType { CSR, DU_VI }

        private IGeometricMultigridModel Model;
        private bool GaussSeidel;
        private bool[] LevelDown;
        private int[] LevelIterations;
        private int firstLevel;
        private int totalLevels;
        private MatrixType matType;

        private Vector b;

        private IMatrixView[] LevelStiffness;
        private LdlSkyline coarseStiffnessLdlFactorized;
        private CsrMatrix[] restriction;
        private CsrMatrix[] interpolation;

        private const int MaxCircleIterations = 10000;

        public int MaxIterations { get; set; }
        public double ConvergenceTolerance { get; set; }

        public int LevelSteps { get => LevelDown.Length; }

        public bool IsStepDown(int i) => LevelDown[i];

        public static GeometricMultigridSolver createSimpleV(IGeometricMultigridModel model, bool GaussSeidel = true, MatrixType matType = MatrixType.CSR,
                                    int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int fineLevelIterations = 4)
        {
            GeometricMultigridSolver a = new GeometricMultigridSolver(model, GaussSeidel);
            a.Initialize(matType, maxCircleIterations, convergenceTolerance, new bool[] { true, false }, new int[] { fineLevelIterations });
            return a;
        }

        public static GeometricMultigridSolver createDeepV(IGeometricMultigridModel model, bool GaussSeidel = true, MatrixType matType = MatrixType.CSR,
                           int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int depth = 2, int levelIterations = 4)
        {
            GeometricMultigridSolver a = new GeometricMultigridSolver(model, GaussSeidel);
            a.Initialize(matType, maxCircleIterations,
                        convergenceTolerance,
                        Enumerable.Repeat(true, depth).Concat(Enumerable.Repeat(false, depth)).ToArray(), new int[] { levelIterations });
            return a;
        }        

        public static GeometricMultigridSolver createDeepV(IGeometricMultigridModel model, int[] levelIterations, bool GaussSeidel = true, MatrixType matType = MatrixType.CSR,
                            int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6)
        {
            GeometricMultigridSolver a = new GeometricMultigridSolver(model, GaussSeidel);
            a.Initialize(matType, maxCircleIterations,
                        convergenceTolerance,
                        Enumerable.Repeat(true, levelIterations.Length / 2).Concat(Enumerable.Repeat(false, levelIterations.Length / 2)).ToArray(), levelIterations);
            return a;
        }

        public GeometricMultigridSolver(IGeometricMultigridModel model, bool GS) { Model = model; GaussSeidel = GS; }

        public void Initialize(MatrixType matType, int maxCircleIterations, double convergenceTolerance, bool[] levelDown, int[] levelIterations)
        {
            this.matType = matType;
            LevelDown = levelDown;
            LevelIterations = levelIterations;
            MaxIterations = maxCircleIterations;
            ConvergenceTolerance = convergenceTolerance;

            // Make the path
            int currentLevel = 0;
            totalLevels = 1;
            for (int i = 0; i < levelDown.Length; ++i)
            {
                if (levelDown[i])
                { 
                    ++currentLevel;
                    if (currentLevel >= totalLevels) totalLevels = currentLevel + 1;
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

            // Generate coarser models
            LevelStiffness = new IMatrixView[totalLevels - 1];
            restriction = new CsrMatrix[totalLevels - 1];
            interpolation = new CsrMatrix[totalLevels - 1];
            (DokRowMajor A, b) = Model.CreateLinearSystem();
            IGeometricMultigridModel currentModel = Model;
            for (int i = 0; i < totalLevels - 1; ++i)
            {
                (currentModel, DokRowMajor restrictionB, DokRowMajor interpolationB) = currentModel.CreateCoarserModelAndSmoothenerMatrices();
                switch(matType)
                {
                    case MatrixType.CSR: LevelStiffness[i] = A.BuildCsrMatrix(true); break;
                    case MatrixType.DU_VI: LevelStiffness[i] = new DuViCompressedSparseMatrix(A); break;
                }
                restriction[i] = restrictionB.BuildCsrMatrix(true);
                interpolation[i] = interpolationB.BuildCsrMatrix(true);
                (A, _) = currentModel.CreateLinearSystem();
            }
            SkylineMatrix coarseStiffness = SkylineMatrix.CreateFromMatrix(A.BuildCsrMatrix(true).CopyToFullMatrix(), 1e-15);
            coarseStiffnessLdlFactorized = coarseStiffness.FactorLdl(true, 1e-15);

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


        private string GetLogPath() => "output_" + (matType == MatrixType.CSR ? "csr_" : "duvi_") + (GaussSeidel ? "gauss_seidel" : "jacobi") + "_cpu.txt";

        private void OutputVectorX(int currentLevel, Vector[] x, string name)
        {
#if DEBUG
            OutputVectorX(currentLevel, x[currentLevel], name);
#endif
        }

        private void OutputVectorX(int currentLevel, Vector x, string name)
        {
#if DEBUG
            string line = name + "(" + currentLevel + ") = [" + string.Join(" ", x.RawData.Select(e => e.ToString("G14"))) + "]" + Environment.NewLine;
            File.AppendAllText(GetLogPath(), line);
#endif
        }

        public (IterativeStatistics, double[]) Solve(Vector? xInitialGuess)
        {
            IStationaryIteration[] stationaryIteration = new IStationaryIteration[totalLevels - 1];
            for (int i = 0; i < stationaryIteration.Length; ++i)
            {
                stationaryIteration[i] = matType == MatrixType.CSR
                    ? GaussSeidel ? new GaussSeidelIterationCsr() : new JacobiIterationCsr()
                    : GaussSeidel ? new GaussSeidelIterationCsrDuVi() : new GaussSeidelIterationCsrDuVi(); //TODO: needs implementation of Jacobi for DuVi matrices
                stationaryIteration[i].UpdateMatrix(LevelStiffness[i], false);
            }

            Vector[] x = new Vector[totalLevels - 1];
            x[0] = xInitialGuess;

            Vector[] r = new Vector[totalLevels];
            r[0] = b;
            // if first level is not 0
            for (int i = 0; i < firstLevel; ++i)
                r[i + 1] = restriction[i].Multiply(r[i]);

            double[] time = new double[totalLevels];
            Stopwatch stopwatch = new Stopwatch();

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
                        Vector eFine = interpolation.Last().Multiply(eCoarse);
                        x[currentLevel - 1].AddIntoThis(eFine);
                        OutputVectorX(currentLevel - 1, x, "X");
                    }
                    else
                    {
                        //TODO: remove me
                        if (currentLevel == 1 && !LevelDown[step]) OutputVectorX(currentLevel, r, "b");


                        // try to solve A * xInitialGuess = b
                        int lvlIter = LevelIterations[Math.Min(currentLevel, LevelIterations.Length - 1)];
                        for (int i = 0; i < lvlIter; ++i)
                        {
                            //TODO: remove me
                            if (currentLevel == 1 && !LevelDown[step]) OutputVectorX(currentLevel, x, "x");
                            stationaryIteration[currentLevel].Execute(r[currentLevel], x[currentLevel]);
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
                                    IterativeStatistics stats = new IterativeStatistics();
                                    stats.NumIterationsRequired = iterations;
                                    stats.ConvergenceCriterion = ("dumb text", residual);
                                    stats.HasConverged = converged;
                                    return (stats, time);
                                }
                            }

                            OutputVectorX(currentLevel, r, "B");
                            OutputVectorX(currentLevel, x, "X");
                            OutputVectorX(currentLevel, rFine, "R");

                            // fine residual to coarse residual
                            r[currentLevel + 1] = restriction[currentLevel].Multiply(rFine);
                            if (currentLevel < totalLevels - 2)
                                x[currentLevel + 1] = Vector.CreateZero(r[currentLevel + 1].Length);
                        }
                        else
                        {
                            Vector eFine = interpolation[currentLevel - 1].Multiply(x[currentLevel]);
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
