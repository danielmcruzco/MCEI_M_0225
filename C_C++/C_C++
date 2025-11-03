
#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd &A, double tol = 1e-6) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto &singularValues = svd.singularValues();
    Eigen::VectorXd singularValuesInv(singularValues.size());

    for (int i = 0; i < singularValues.size(); ++i) {
        singularValuesInv(i) = (singularValues(i) > tol) ? 1.0 / singularValues(i) : 0;
    }

    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}

int main() {
    Eigen::MatrixXd A(4, 3);
    A << 1, 0, 2,
         2, -1, 5,
         0, 1, -1,
         1, 3, -1;

    std::cout << "Matriz A:\n" << A << "\n";

    Eigen::MatrixXd A_pinv = pseudoInverse(A);

    std::cout << "Pseudoinversa de A (Moore-Penrose):\n" << A_pinv << "\n";

    return 0;
}
