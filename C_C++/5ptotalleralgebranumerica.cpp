
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

int main() {
    // Datos experimentales
    Eigen::VectorXd x(5), y(5);
    x << 0.0, 0.5, 1.0, 1.5, 2.0;
    y << 1.0, 2.2, 2.8, 3.6, 5.1;

    int n = x.size();

    // ---------------------------
    // Interpolación polinómica (grado 4)
    // ---------------------------
    int grado_interp = n - 1;
    Eigen::MatrixXd V_interp(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            V_interp(i, j) = std::pow(x(i), grado_interp - j);
        }
    }
    Eigen::VectorXd coef_interp = V_interp.fullPivLu().solve(y);

    std::cout << "Coeficientes interpolación polinómica (grado 4):\n" << coef_interp.transpose() << "\n";

    // Evaluar interpolación
    Eigen::VectorXd y_interp = V_interp * coef_interp;

    // Calcular ECM interpolación
    double ecm_interp = (y - y_interp).squaredNorm() / n;
    std::cout << "ECM interpolación: " << ecm_interp << "\n\n";

    // ---------------------------
    // Ajuste por mínimos cuadrados (grado 2) usando SVD
    // ---------------------------
    int grado_svd = 2;
    Eigen::MatrixXd V_svd(n, grado_svd + 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= grado_svd; ++j) {
            V_svd(i, j) = std::pow(x(i), grado_svd - j);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V_svd, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd coef_svd = svd.solve(y);

    std::cout << "Coeficientes ajuste por mínimos cuadrados (grado 2):\n" << coef_svd.transpose() << "\n";

    // Evaluar ajuste SVD
    Eigen::VectorXd y_svd = V_svd * coef_svd;

    // Calcular ECM ajuste SVD
    double ecm_svd = (y - y_svd).squaredNorm() / n;
    std::cout << "ECM ajuste SVD: " << ecm_svd << "\n";

    return 0;
}
