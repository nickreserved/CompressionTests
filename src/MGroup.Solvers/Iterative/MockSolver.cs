namespace MGroup.Solvers.Iterative
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	using MGroup.MSolve.Solution;
	using MGroup.MSolve.Solution.LinearSystem;
	using MGroup.Solvers.LinearAlgebraExtensions.Matrices.Builders;
	using MGroup.Solvers.LinearSystem;

	public class MockSolver : ISolver
	{
		private readonly GlobalAlgebraicModel<DokRowMajor> algebraicModel;

		public MockSolver(GlobalAlgebraicModel<DokRowMajor> algebraicModel)
		{
			this.algebraicModel = algebraicModel;
			LinearSystem = algebraicModel.LinearSystem;
		}

		public IGlobalLinearSystem LinearSystem { get; }

		public ISolverLogger Logger => throw new NotImplementedException();

		public string Name => "";

		public void HandleMatrixWillBeSet() { }
		public void Initialize() { }
		public void PreventFromOverwrittingSystemMatrices() { }
		public void Solve() => throw new NotImplementedException();
	}
}
