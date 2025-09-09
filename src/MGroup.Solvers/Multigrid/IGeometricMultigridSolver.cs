using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Vectors;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public interface IGeometricMultigridSolver
    {
        (IterativeStatistics, double[]) Solve(Vector? xInitialGuess);
    }
}
