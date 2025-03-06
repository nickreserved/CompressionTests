using Compression.src.MGroup.LinearAlgebra.Iterative.Stationary;
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
    public class GeometricMultigridSolver
    {
        public enum MatrixType { CSR, DU_VI }

        private IGeometricMultigridModel Model;
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

        public static GeometricMultigridSolver createSimpleV(IGeometricMultigridModel model, MatrixType matType = MatrixType.CSR,
                                    int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int fineLevelIterations = 4)
        {
            GeometricMultigridSolver a = new GeometricMultigridSolver(model);
            a.Initialize(maxCircleIterations, convergenceTolerance, new bool[] { true, false }, new int[] { fineLevelIterations }, matType);
            return a;
        }

        public static GeometricMultigridSolver createDeepV(IGeometricMultigridModel model, MatrixType matType = MatrixType.CSR,
                           int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int depth = 2, int levelIterations = 4)
        {
            GeometricMultigridSolver a = new GeometricMultigridSolver(model);
            a.Initialize(maxCircleIterations, convergenceTolerance,
                        Enumerable.Repeat(true, depth).Concat(Enumerable.Repeat(false, depth)).ToArray(),
                        new int[] { levelIterations }, matType);
            return a;
        }        

        public static GeometricMultigridSolver createDeepV(IGeometricMultigridModel model, int[] levelIterations, MatrixType matType = MatrixType.CSR,
                            int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6)
        {
            GeometricMultigridSolver a = new GeometricMultigridSolver(model);
            a.Initialize(maxCircleIterations, convergenceTolerance,
                        Enumerable.Repeat(true, levelIterations.Length / 2).Concat(Enumerable.Repeat(false, levelIterations.Length / 2)).ToArray(),
                        levelIterations, matType);
            return a;
        }

        public GeometricMultigridSolver(IGeometricMultigridModel model) => Model = model;
        public void Initialize(int maxCircleIterations, double convergenceTolerance, bool[] levelDown, int[] levelIterations, MatrixType matType)
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
        }

        public (IterativeStatistics, double[]) Solve(Vector xInitialGuess)
        {
            IStationaryIteration[] methodGaussSeidel = new IStationaryIteration[totalLevels - 1];
            for (int i = 0; i < methodGaussSeidel.Length; ++i)
            {
                methodGaussSeidel[i] = matType == MatrixType.CSR
                    ? new GaussSeidelIterationCsr()
                    : new GaussSeidelIterationCsrDuVi();
                methodGaussSeidel[i].UpdateMatrix(LevelStiffness[i], false);
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
                        Vector eCoarse = coarseStiffnessLdlFactorized.SolveLinearSystem(r[currentLevel]);
                        Vector eFine = interpolation.Last().Multiply(eCoarse);
                        x[currentLevel - 1].AddIntoThis(eFine);
                    }
                    else
                    {
                        // try to solve A * xInitialGuess = b
                        int lvlIter = LevelIterations[Math.Min(currentLevel, LevelIterations.Length - 1)];
                        for (int i = 0; i < lvlIter; ++i)
                            methodGaussSeidel[currentLevel].Execute(r[currentLevel], x[currentLevel]);

                        if (LevelDown[step])
                        {
                            // calculate residual
                            Vector rFine = r[currentLevel].Copy();
                            rFine.SubtractIntoThis(LevelStiffness[currentLevel].Multiply(x[currentLevel]));

                            // small residual or exceeded the iteration number
                            if (currentLevel == 0)
                            {
                                double residual = rFine.Norm2();
                                bool converged = residual <= ConvergenceTolerance;
                                if (converged || iterations > MaxIterations)
                                {
                                    IterativeStatistics stats = new IterativeStatistics();
                                    stats.NumIterationsRequired = iterations;
                                    stats.ConvergenceCriterion = ("dumb text", residual);
                                    stats.HasConverged = converged;
                                    return (stats, time);
                                }
                            }

                            // fine residual to coarse residual
                            r[currentLevel + 1] = restriction[currentLevel].Multiply(rFine);
                            if (currentLevel < totalLevels - 2)
                                x[currentLevel + 1] = Vector.CreateZero(r[currentLevel + 1].Length);
                        }
                        else
                        {
                            Vector eFine = interpolation[currentLevel - 1].Multiply(x[currentLevel]);
                            x[currentLevel - 1].AddIntoThis(eFine);
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
