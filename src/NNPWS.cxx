#include <iostream>
#include "ModelLoader.hxx"

int main() {
    // ==========================================
    // 1. INITIALISATION (Une seule fois au démarrage)
    // ==========================================
    std::string path = "unified_water_model.pt";

    if (!ModelLoader::instance().load(path)) {
        std::cerr << "Arrêt du programme : échec du chargement du modèle." << std::endl;
        return -1;
    }

    /*
    // ==========================================
    // 2. UTILISATION (N'importe où dans le code)
    // ==========================================

    // Création d'un calculateur léger
    GibbsPredictor calculator;

    // Test 1 : Liquide
    double t1 = 300.0;
    double p1 = 10.0;
    double res1 = calculator.compute_gibbs(t1, p1);
    std::cout << "G(" << t1 << "K, " << p1 << "MPa) = " << res1 << std::endl;

    // Test 2 : Vapeur
    double t2 = 800.0;
    double p2 = 5.0;
    double res2 = calculator.compute_gibbs(t2, p2);
    std::cout << "G(" << t2 << "K, " << p2 << "MPa) = " << res2 << std::endl;


    */
    return 0;
}