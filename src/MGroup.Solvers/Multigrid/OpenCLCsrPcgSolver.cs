using CASS.OpenCL;
using CASS.Types;
using Compression.src.MGroup.OCL;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Triangulation;
using MGroup.LinearAlgebra.Vectors;
using MGroup.OCL;
using System.Diagnostics;
using System.Xml.Linq;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public class OpenCLCsrPcgSolver
    {
        private int DoFs;
        private SizeT[] GlobalWorkSize;
        private SizeT[] LocalWorkSize;

        private readonly OpenCL context;
        private CLProgram program;

        private CLKernel kernelDot1stPass, kernelDot2ndPass, kernelDot2ndPassCalcA, kernelUpdateXRZandCheckResidual, kernelDot2ndPassCalcB, kernelUpdateP, kernelMatrixVectorProduct, kernelInit0, kernelInit;

        private CLCommandQueue commandQueue;

        private CLMem bufferOfRowOffsets, bufferOfColumnIndices, bufferOfValues;
        private CLMem bufferOfVectorP, bufferOfVectorZ, bufferOfVectorR, bufferOfVectorX, bufferOfVectorAP, bufferOfJacobiPreconditioner;
        private CLMem bufferOfScalars, bufferOfZero;
        private CLMem bufferOfIntermediatePartialDot;

        public int MaxIterations { get; set; }
        public double ConvergenceTolerance { get; set; }

        /// <summary>
        /// Constructor of solver for OpenCL.
        /// </summary>
        /// <param name="context">OpenCL context.</param>
        /// <param name="maxIterations">Number of iterations for the algorithm.</param>
        /// <param name="convergenceTolerance">The tolerance of solution in order to be the problem solved.</param>
        public OpenCLCsrPcgSolver(OpenCL context, int maxIterations, double convergenceTolerance)
        {
            this.context = context;
            MaxIterations = maxIterations;
            ConvergenceTolerance = convergenceTolerance;
        }

        public void ReleaseOpenCLResources()
        {
            context.ReleaseMemObject(bufferOfVectorP);
            context.ReleaseMemObject(bufferOfVectorZ);
            context.ReleaseMemObject(bufferOfVectorR);
            context.ReleaseMemObject(bufferOfVectorX);
            context.ReleaseMemObject(bufferOfVectorAP);
            context.ReleaseMemObject(bufferOfJacobiPreconditioner);
            
            context.ReleaseMemObject(bufferOfRowOffsets);
            context.ReleaseMemObject(bufferOfColumnIndices);
            context.ReleaseMemObject(bufferOfValues);

            context.ReleaseMemObject(bufferOfScalars);
            context.ReleaseMemObject(bufferOfZero);
            context.ReleaseMemObject(bufferOfIntermediatePartialDot);

            context.ReleaseKernel(kernelDot1stPass);
            context.ReleaseKernel(kernelDot2ndPass);
            context.ReleaseKernel(kernelDot2ndPassCalcA);
            context.ReleaseKernel(kernelUpdateXRZandCheckResidual);
            context.ReleaseKernel(kernelDot2ndPassCalcB);
            context.ReleaseKernel(kernelUpdateP);
            context.ReleaseKernel(kernelMatrixVectorProduct);
            context.ReleaseKernel(kernelInit0);
            context.ReleaseKernel(kernelInit);

            context.ReleaseCommandQueue(commandQueue);

            context.ReleaseProgram(program);
       }
        
        /// <summary>
        /// Initialize Geometric Multigrid solver
        /// </summary>
        /// <param name="device">OpenCL device.</param>
        /// <param name="model">The geometric multigrid model.</param>
        public void Initialize(Device device, IGeometricMultigridModel model)
        {
            // Generate model
            (DokRowMajor AA, Vector b) = model.CreateLinearSystem();
            CsrMatrix A = AA.BuildCsrMatrix(true);
            Vector preconditioner = GeometricMultigridSolver.JacobiPreconditioner(AA.RawRows);
            DoFs = A.NumRows;

            // initialize number of global and local work-items for kernels
            uint LocalWorkgroupSize = Math.Min(device.workgroupSizeMax, (uint)device.workItemSizes[0]);
            LocalWorkSize = new SizeT[1];
            GlobalWorkSize = new SizeT[1];
            int numWorkGroups;
            const bool NonUniformWorkgroup = false;
            if (DoFs <= LocalWorkgroupSize) { GlobalWorkSize[0] = LocalWorkSize[0] = DoFs; numWorkGroups = 1; }
            else
            {
                SizeT l = LocalWorkSize[0] = LocalWorkgroupSize;
                GlobalWorkSize[0] = DoFs;
                numWorkGroups = ((DoFs + l - 1) / l);
                if (!NonUniformWorkgroup) GlobalWorkSize[0] = numWorkGroups * l;
            }

            // Initialize OpenCL
            program = Program.CreateProgram(context, "CsrCG", "-cl-std=CL2.0");

            kernelDot1stPass                = context.CreateKernel(program, "dot_partial");
            kernelDot2ndPass                = context.CreateKernel(program, "dot_finalize");
            kernelDot2ndPassCalcA           = context.CreateKernel(program, "dot_finalize_and_calc_a");
            kernelUpdateXRZandCheckResidual = context.CreateKernel(program, "update_x_r_z_check_r");
            kernelDot2ndPassCalcB           = context.CreateKernel(program, "dot_finalize_and_calc_b");
            kernelUpdateP                   = context.CreateKernel(program, "update_p");
            kernelMatrixVectorProduct       = context.CreateKernel(program, "matrix_vector_product");
            kernelInit0                     = context.CreateKernel(program, "initialize0");
            kernelInit                      = context.CreateKernel(program, "initialize");

            commandQueue = context.CreateCommandQueue(context.Devices[0]);


            // Reserve and write OpenCL buffers
            // ... for matrix
            bufferOfRowOffsets    = context.CreateBuffer(CLMemFlags.ReadOnly,        A.RawRowOffsets.Length * sizeof(UInt32));
            bufferOfColumnIndices = context.CreateBuffer(CLMemFlags.ReadOnly,        A.RawColIndices.Length * sizeof(UInt32));
            bufferOfValues        = context.CreateBuffer(CLMemFlags.ReadOnly,        A.RawValues.Length     * sizeof(double));
            context.WriteBuffer(commandQueue, bufferOfRowOffsets,    CLBool.True, 0, A.RawRowOffsets.Length * sizeof(UInt32), A.RawRowOffsets);
            context.WriteBuffer(commandQueue, bufferOfColumnIndices, CLBool.True, 0, A.RawColIndices.Length * sizeof(UInt32), A.RawColIndices);
            context.WriteBuffer(commandQueue, bufferOfValues,        CLBool.True, 0, A.RawValues.Length     * sizeof(double), A.RawValues);
            // ...for vectors
            bufferOfVectorP                = context.CreateBuffer(CLMemFlags.ReadWrite,          DoFs * sizeof(double));
            bufferOfVectorZ                = context.CreateBuffer(CLMemFlags.ReadWrite,          DoFs * sizeof(double));
            bufferOfVectorR                = context.CreateBuffer(CLMemFlags.ReadWrite,          DoFs * sizeof(double));
            bufferOfVectorX                = context.CreateBuffer(CLMemFlags.ReadWrite,          DoFs * sizeof(double));
            bufferOfVectorAP               = context.CreateBuffer(CLMemFlags.ReadWrite,          DoFs * sizeof(double));
            bufferOfJacobiPreconditioner   = context.CreateBuffer(CLMemFlags.ReadOnly,           DoFs * sizeof(double));
            bufferOfIntermediatePartialDot = context.CreateBuffer(CLMemFlags.ReadWrite, numWorkGroups * sizeof(double));
            context.WriteBuffer(commandQueue, bufferOfVectorR,              CLBool.True, 0,      DoFs * sizeof(double), b.RawData);
            context.WriteBuffer(commandQueue, bufferOfJacobiPreconditioner, CLBool.True, 0,      DoFs * sizeof(double), preconditioner.RawData);
            // ...for scalars
            bufferOfZero    = context.CreateBuffer(CLMemFlags.WriteOnly, 2 * sizeof(UInt32));
            bufferOfScalars = context.CreateBuffer(CLMemFlags.ReadWrite, 2 * sizeof(double));

            // If we want NON-BLOCKING transfer we can use CLBool.False and then after each context.WriteBuffer we must keep the context.LastEnqueueEvent
            // then we must use clWaitForEvents(num, events) or clGetEventInfo(...) or clSetEventCallback(event, ...)
            // Also clNDRangeKernel() support previous events to be executed -- unfortunatelly not the wrapper version but the C version underneath
            // It starts to be overcomplicated for a test

            // Now the parameters for each kernel follows.
            // What is commented out, takes different params while algorithm executed.
            // This happens on solve()

            //context.SetKernelArg(kernelDot1stPass, 0, bufferOfVectorR OR bufferOfVectorP);
            //context.SetKernelArg(kernelDot1stPass, 1, bufferOfVectorZ OR bufferOfVectorAP);
            context.SetKernelArg(kernelDot1stPass, 2, bufferOfIntermediatePartialDot);
            OpenCLDuViGeometricMultigridSolver.ThrowCLException(OpenCLDriver.clSetKernelArg(
                                 kernelDot1stPass, 3, LocalWorkSize[0] * sizeof(double), 0));
            context.SetKernelArg(kernelDot1stPass, 4, DoFs);

            context.SetKernelArg(kernelDot2ndPass, 0, bufferOfIntermediatePartialDot);
            context.SetKernelArg(kernelDot2ndPass, 1, bufferOfScalars);
            OpenCLDuViGeometricMultigridSolver.ThrowCLException(OpenCLDriver.clSetKernelArg(
                                 kernelDot2ndPass, 2, LocalWorkSize[0] * sizeof(double), 0));
            context.SetKernelArg(kernelDot2ndPass, 3, numWorkGroups);

            context.SetKernelArg(kernelDot2ndPassCalcA, 0, bufferOfIntermediatePartialDot);
            context.SetKernelArg(kernelDot2ndPassCalcA, 1, bufferOfScalars);
            OpenCLDuViGeometricMultigridSolver.ThrowCLException(OpenCLDriver.clSetKernelArg(
                                 kernelDot2ndPassCalcA, 2, LocalWorkSize[0] * sizeof(double), 0));
            context.SetKernelArg(kernelDot2ndPassCalcA, 3, numWorkGroups);

            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 0, bufferOfJacobiPreconditioner);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 1, bufferOfVectorAP);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 2, bufferOfVectorP);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 3, bufferOfScalars);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 4, bufferOfVectorX);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 5, bufferOfVectorR);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 6, bufferOfVectorZ);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 7, DoFs);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 8, bufferOfZero);
            context.SetKernelArg(kernelUpdateXRZandCheckResidual, 9, ConvergenceTolerance);

            context.SetKernelArg(kernelDot2ndPassCalcB, 0, bufferOfIntermediatePartialDot);
            context.SetKernelArg(kernelDot2ndPassCalcB, 1, bufferOfScalars);
            OpenCLDuViGeometricMultigridSolver.ThrowCLException(OpenCLDriver.clSetKernelArg(
                                 kernelDot2ndPassCalcB, 2, LocalWorkSize[0] * sizeof(double), 0));
            context.SetKernelArg(kernelDot2ndPassCalcB, 3, numWorkGroups);

            context.SetKernelArg(kernelUpdateP, 0, bufferOfVectorZ);
            context.SetKernelArg(kernelUpdateP, 1, bufferOfVectorP);
            context.SetKernelArg(kernelUpdateP, 2, bufferOfScalars);
            context.SetKernelArg(kernelUpdateP, 3, DoFs);

            context.SetKernelArg(kernelMatrixVectorProduct, 0, bufferOfRowOffsets);
            context.SetKernelArg(kernelMatrixVectorProduct, 1, bufferOfColumnIndices);
            context.SetKernelArg(kernelMatrixVectorProduct, 2, bufferOfValues);
            context.SetKernelArg(kernelMatrixVectorProduct, 3, bufferOfVectorP);
            context.SetKernelArg(kernelMatrixVectorProduct, 4, bufferOfVectorAP);
            context.SetKernelArg(kernelMatrixVectorProduct, 5, (byte) 1);
            context.SetKernelArg(kernelMatrixVectorProduct, 6, DoFs);

            context.SetKernelArg(kernelInit0, 0, bufferOfJacobiPreconditioner);
            context.SetKernelArg(kernelInit0, 1, bufferOfVectorR);
            context.SetKernelArg(kernelInit0, 2, bufferOfVectorP);
            context.SetKernelArg(kernelInit0, 3, bufferOfVectorZ);
            context.SetKernelArg(kernelInit0, 4, DoFs);

            context.SetKernelArg(kernelInit, 0, bufferOfRowOffsets);
            context.SetKernelArg(kernelInit, 1, bufferOfColumnIndices);
            context.SetKernelArg(kernelInit, 2, bufferOfValues);
            context.SetKernelArg(kernelInit, 3, bufferOfJacobiPreconditioner);
            context.SetKernelArg(kernelInit, 4, bufferOfVectorX);
            context.SetKernelArg(kernelInit, 5, bufferOfVectorR);
            context.SetKernelArg(kernelInit, 6, bufferOfVectorP);
            context.SetKernelArg(kernelInit, 7, bufferOfVectorZ);
            context.SetKernelArg(kernelInit, 8, DoFs);
        }

        private void OutputVector(CLMem bufferOfVector, string name) => OutputVector(bufferOfVector, name, context, commandQueue, DoFs);

        private static void OutputVector(CLMem bufferOfVector, string name, OpenCL context, CLCommandQueue commandQueue, int DoFs)
        {
            #if DEBUG
            Vector x = Vector.CreateZero(DoFs);
            context.ReadBuffer(commandQueue, bufferOfVector, CLBool.True, 0, x.Length * sizeof(double), x.RawData);
            string line = name + " = [" + string.Join(" ", x.RawData.Select(e => e.ToString("G14"))) + "]" + Environment.NewLine;
            File.AppendAllText("output_csr_cg_gpu.txt", line);
            #endif
        }

        private void OutputScalar(CLMem bufferOfVector, int index, string name) => OutputScalar(bufferOfVector, index, name, context, commandQueue);

        private static void OutputScalar(CLMem bufferOfVector, int index, string name, OpenCL context, CLCommandQueue commandQueue)
        {
            #if DEBUG
            double[] x = new double[1];
            context.ReadBuffer(commandQueue, bufferOfVector, CLBool.True, index * sizeof(double), sizeof(double), x);
            string line = name + " = " + x[0] + Environment.NewLine;
            File.AppendAllText("output_csr_cg_gpu.txt", line);
            #endif
        }

        /// <summary>
        /// Solves the geometric multigrid.
        /// </summary>
        /// <param name="xInitialGuess">The initial guess x vector. If initially first level is not 0, then it corresponds to that level (<see cref="firstLevel"/>).
        /// After solving, it has the solution if the problem converges.</param>
        /// <returns>Algorithm statistics</returns>
        public (Vector, IterativeStatistics) Solve(Vector? xInitialGuess)
        {
            if (DoFs == 0) throw new InvalidOperationException("You must call Initialize(model) first");

            UInt32[] con = new UInt32[1]; // convergence check buffer on host
            
            // initialization
            if (xInitialGuess != null)
            {
                Debug.Assert(xInitialGuess.Length == DoFs);
                context.WriteBuffer(commandQueue, bufferOfVectorX, CLBool.True, 0, DoFs * sizeof(double), xInitialGuess.RawData);
                context.NDRangeKernel(commandQueue, kernelInit, 1, null, GlobalWorkSize, LocalWorkSize);
            }
            else
            {
                xInitialGuess = Vector.CreateZero(DoFs);
                context.FillBuffer(commandQueue, bufferOfVectorX, 0, DoFs * sizeof(double), 0.0);
                context.NDRangeKernel(commandQueue, kernelInit0, 1, null, GlobalWorkSize, LocalWorkSize);
            }

            OutputVector(bufferOfVectorX, "X");
            OutputVector(bufferOfVectorR, "R");
            OutputVector(bufferOfVectorP, "P");
            OutputVector(bufferOfVectorZ, "Z");

            context.SetKernelArg(kernelDot1stPass, 0, bufferOfVectorR);
            context.SetKernelArg(kernelDot1stPass, 1, bufferOfVectorZ);
            context.NDRangeKernel(commandQueue, kernelDot1stPass, 1, null, GlobalWorkSize, LocalWorkSize);  // needs cure

            context.NDRangeKernel(commandQueue, kernelDot2ndPass, 1, null, LocalWorkSize, LocalWorkSize);

            // loop algorithm
            for (int currentIteration = 0; ; ++currentIteration)
            {
                OutputScalar(bufferOfScalars, 0, "r*z");

                context.NDRangeKernel(commandQueue, kernelMatrixVectorProduct, 1, null, GlobalWorkSize, LocalWorkSize);
                OutputVector(bufferOfVectorAP, "AP");

                context.SetKernelArg(kernelDot1stPass, 0, bufferOfVectorP);
                context.SetKernelArg(kernelDot1stPass, 1, bufferOfVectorAP);
                context.NDRangeKernel(commandQueue, kernelDot1stPass, 1, null, GlobalWorkSize, LocalWorkSize);  // needs cure
                
                context.NDRangeKernel(commandQueue, kernelDot2ndPassCalcA, 1, null, LocalWorkSize, LocalWorkSize);

                OutputScalar(bufferOfScalars, 1, "a");

                context.FillBuffer(commandQueue, bufferOfZero, 0, sizeof(UInt32), 1);   // converged
                context.NDRangeKernel(commandQueue, kernelUpdateXRZandCheckResidual, 1, null, GlobalWorkSize, LocalWorkSize);

                OutputVector(bufferOfVectorX, "X");
                OutputVector(bufferOfVectorR, "R");
                OutputVector(bufferOfVectorZ, "Z");

                context.ReadBuffer(commandQueue, bufferOfZero, CLBool.True, 0, sizeof(UInt32), con);
                bool converged = con[0] == 1;
                bool failed = (con[0] & 2) == 2;  // some numbers become NaN
                // small residual or exceeded the iteration number
                if (converged || failed || currentIteration > MaxIterations)
                {
                    context.ReadBuffer(commandQueue, bufferOfVectorX, CLBool.True, 0, xInitialGuess!.Length * sizeof(double), xInitialGuess.RawData);
                    return (xInitialGuess, new IterativeStatistics
                    {
                        NumIterationsRequired = currentIteration,
                        ConvergenceCriterion = ("dumb text", ConvergenceTolerance),
                        HasConverged = converged
                    });
                }

                context.SetKernelArg(kernelDot1stPass, 0, bufferOfVectorR);
                context.SetKernelArg(kernelDot1stPass, 1, bufferOfVectorZ);
                context.NDRangeKernel(commandQueue, kernelDot1stPass, 1, null, GlobalWorkSize, LocalWorkSize);  // needs cure

                context.NDRangeKernel(commandQueue, kernelDot2ndPassCalcB, 1, null, LocalWorkSize, LocalWorkSize);

                OutputScalar(bufferOfScalars, 1, "b");

                context.NDRangeKernel(commandQueue, kernelUpdateP, 1, null, GlobalWorkSize, LocalWorkSize);

                OutputVector(bufferOfVectorP, "P");
            }
        }
    }
}
