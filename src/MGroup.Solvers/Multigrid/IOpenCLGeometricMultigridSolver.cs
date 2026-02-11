using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    internal interface IOpenCLGeometricMultigridSolver : IGeometricMultigridSolver
    {
        void ReleaseOpenCLResources();
    }
}
