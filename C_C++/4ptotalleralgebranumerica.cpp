
#include <iostream>
#include <Eigen/Dense>

int main() {

    // 4. Inversa por tres métodos
    Eigen::MatrixXd F(3, 3);
    F << 2, 1, 1,
         1, 3, 2,
         1, 0, 0;

    std::cout << "Matriz F:\n" << F << "\n\n";

    // Método directo
    Eigen::MatrixXd F_inv_direct = F.inverse();
    std::cout << "Inversa por método directo:\n" << F_inv_direct << "\n\n";

    // QR
    Eigen::HouseholderQR<Eigen::MatrixXd> qrF(F);
    Eigen::MatrixXd QF = qrF.householderQ();
    Eigen::MatrixXd RF = qrF.matrixQR().triangularView<Eigen::Upper>();
    Eigen::MatrixXd F_inv_QR = RF.inverse() * QF.transpose();
    std::cout << "Inversa por QR:\n" << F_inv_QR << "\n\n";

    // SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd singularValuesInv = svdF.singularValues();
    for (int i = 0; i < singularValuesInv.size(); ++i) {
        singularValuesInv(i) = (singularValuesInv(i) > 1e-6) ? 1.0 / singularValuesInv(i) : 0;
    }
    Eigen::MatrixXd F_inv_SVD = svdF.matrixV() * singularValuesInv.asDiagonal() * svdF.matrixU().transpose();
    std::cout << "Inversa por SVD:\n" << F_inv_SVD << "\n";

    return 0;
}
