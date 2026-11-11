namespace Compression.src.MGroup.Solvers.Multigrid
{
    internal interface IOpenCLGeometricMultigridSolver : IGeometricMultigridSolver
    {
        void ReleaseOpenCLResources();
    }
}
