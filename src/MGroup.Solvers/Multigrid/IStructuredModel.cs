using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Meshes.Structured;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public interface IStructuredModel
    {
        /// <summary>
        /// The structured mesh of the model.
        /// </summary>
        public IStructuredMesh Mesh { get; }

        /// <summary>
        /// Number of DoFs in model.
        /// </summary>
        public int NumDofsAll { get; }
        /// <summary>
        /// Number of free DoFs in model.
        /// </summary>
        public int NumDofsFree { get; }

        /// <summary>
        /// It checks DoF with id <paramref name="dof"/>, if it is free or fixed.
        /// </summary>
        /// <param name="dof">The id of DoF. Probably the nodeID = dof / NumDofsPerNode, and the dofOfNode = dof % NumDofsPerNode</param>
        /// <returns>True if DoF is free, false if it is fixed.</returns>
        public bool IsDofFree(int dof);

        /// <summary>
        /// Generates a more detailed or more simpler model.
        /// </summary>
        /// <param name="detail">How much detailed or simpler model will be produced in terms of power of 2. If <paramref name="detail"/> is positive
        /// e.g. 1, then a detailed model will be produced with double (2^1) model length in every dimension. If <paramref name="detail"/> is
        /// negative, e.g. -2 a coarser model will be produced with one-quarter (2^-2) model length in every dimension.</param>
        /// <returns>The generated model, either detailed-finer or simpler-coarser.</returns>
        public IStructuredModel GenerateModel(int detail);

        /// <summary>
        /// For the current model, it creates the static linear system consisted from stiffness matrix A and load vector b.
        /// </summary>
        /// <returns>First element of tuple is the stiffness matrix of model, which corresponds to free DoFs.
        /// Second element of tuple is the load vector of </returns>
        public (DokRowMajor A, Vector b) CreateLinearSystem();

        /// <summary>
        /// The IDs if the free DoFs in model. They correspond with rows and columns of the stiffness matrix.
        /// </summary>
        /// <returns>An array with free DoFs IDs.</returns>
        public int[] FindFreeDofs();
    }
}
