using CASS.OpenCL;
using CASS.Types;
using Compression.src.MGroup.OCL;
using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Matrices.Operators;
using MGroup.LinearAlgebra.Vectors;
using MGroup.OCL;
using Microsoft.VisualStudio.TestPlatform.ObjectModel;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace Compression.tests.MGroup.Solvers.Tests
{
    public class GeometricMultigridTestsWithOpenCL
    {
         [Fact]
        public void RunMe()
        {
            Platform[] platforms = Platform.GetPlatforms();
            Assert.NotEmpty(platforms);
            Device[] devices = platforms[0].GetDevices();     // select a platform to get devices
            Assert.NotEmpty(devices);
            OpenCL context = new OpenCL(platforms[0].platformId, /*devices[0].deviceId*/devices.Select(x => x.deviceId).ToArray());
            CLProgram program = Program.CreateProgram(context, "HybridGaussSeidel");
            CLKernel kernel = context.CreateKernel(program, "hybrid_gauss_seidel_step_with_CSR");
            CLCommandQueue commandQueue = context.CreateCommandQueue(context.Devices[0]);
            CLMem bufferOfRowIndices = context.CreateBuffer(CLMemFlags.ReadOnly, 1000 * sizeof(double));
            CLMem bufferOfColumnIndices = context.CreateBuffer(CLMemFlags.ReadOnly, 1000 * sizeof(double));
            CLMem bufferOfValues = context.CreateBuffer(CLMemFlags.ReadOnly, 1000 * sizeof(double));
            CLMem bufferOfVector = context.CreateBuffer(CLMemFlags.ReadWrite, 1000 * sizeof(double));
            context.WriteBuffer<double>(commandQueue, bufferOfRowIndices, CLBool.False, 0, 0, null);
            context.WriteBuffer<double>(commandQueue, bufferOfColumnIndices, CLBool.False, 0, 0, null);
            context.WriteBuffer<double>(commandQueue, bufferOfValues, CLBool.False, 0, 0, null);
            context.WriteBuffer<double>(commandQueue, bufferOfVector, CLBool.False, 0, 0, null);
            context.SetKernelArg(kernel, 0, bufferOfRowIndices);
            context.SetKernelArg(kernel, 1, bufferOfColumnIndices);
            context.SetKernelArg(kernel, 2, bufferOfValues);
            context.SetKernelArg(kernel, 3, bufferOfVector);
            context.SetKernelArg(kernel, 4, new double[100]);
            context.NDRangeKernel(commandQueue, kernel, 2, new SizeT[] { 0 }, new SizeT[] { 5 }, new SizeT[] { 10 });

            Assert.True(true);
        }

        [Fact]
        public static void CheckCantilever2dSolutionDeepVWithOpenCL()
        {
            double convergenceTolerance = 1e-5;
            int iterations = 10000;
            String logFilePath = "out.txt";

            IGeometricMultigridModel model = new FemCantilever2D(new int[] { 256, 16 }, new double[] { 20, 1, 1 });

            Stopwatch stopwatch = new Stopwatch();




            stopwatch.Restart();
            GeometricMultigridSolver solver = GeometricMultigridSolver.createDeepV(model, GeometricMultigridSolver.MatrixType.CSR, iterations, convergenceTolerance, 2, 4);
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            (IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(logFilePath, $"\n\n\n\n\nRequired time for Geometric Multigrid (CSR): {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);





            stopwatch.Restart();
            solver = GeometricMultigridSolver.createDeepV(model, GeometricMultigridSolver.MatrixType.DU_VI, iterations, convergenceTolerance, 2, 4);
            stopwatch.Stop();
            timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            (stats, time) = solver.Solve(x);
            stopwatch.Stop();
            timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(logFilePath, $"\n\n\n\n\nRequired time for Geometric Multigrid (CSR + DuVi): {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);





            stopwatch.Restart();
            var cg = ConjugateGradientInitialize(model, iterations, convergenceTolerance);
            stopwatch.Stop();
            double timeCGI = stopwatch.Elapsed.TotalMilliseconds;
            x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            stats = ConjugateGradientSolve(cg, x);
            stopwatch.Stop();
            double timeCG = stopwatch.Elapsed.TotalMilliseconds;
            File.AppendAllText(logFilePath, $"\n\nRequired time for CG: {timeCGI + timeCG}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeCGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeCG}ms");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
        }
    }
}
