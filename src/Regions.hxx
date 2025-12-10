#ifndef NNPWS_REGIONS_HXX
#define NNPWS_REGIONS_HXX

#include <cmath>

enum Region {
    out_of_regions = -1,
    r1 = 1,
    r2 = 2,
    r3 = 3,
    r4 = 4,
    r5 = 5
};

class Regions_Boundaries {
public:
    // Constantes critiques
    static constexpr double Tc = 647.096; // Kelvin
    static constexpr double Pc = 22.064;  // MPa

    // Équation de la pression de saturation (Psat) en fonction de T
    // Equation region 4 (Frontière entre Région 1 et 2)
    static double saturation_pressure(double T) {
        // Coefficients officiels IAPWS-97 pour P_sat
        constexpr double n[] = {
            0.0,
            0.11670521452767e4,
            -0.72421316703206e6,
            -0.17073846940092e2,
            0.12020824702470e5,
            -0.32325550322333e7,
            0.14915108613530e2,
            -0.48232657361591e4,
            0.40511340542057e6,
            -0.23855557567849,
            0.65017534844798e3
        };

        double v = T + n[9] / (T - n[10]);
        double A = v * v + n[1] * v + n[2];
        double B = n[3] * v * v + n[4] * v + n[5];
        double C = n[6] * v * v + n[7] * v + n[8];

        double term = 2.0 * C / (-B + std::sqrt(B * B - 4.0 * A * C));
        return std::pow(term, 4.0);
    }

    // Équation de la frontière entre Région 2 et 3
    // Equation region 5
    static double boundary_2_3_pressure(double T) {
        constexpr double n[] = {
            0.0,
            0.34805185628969e3,
            -0.11671859879975e1,
            0.10192970039326e-2
        };
        return n[1] + n[2] * T + n[3] * T * T;
    }

    // Déterminer la région
    static Region determine_region(double T, double P) {
        if (T < 273.15) return out_of_regions;
        if (T > 1073.15) return r5; // Haute température (Région 5)
        if (P > 100.0) return out_of_regions;

        if (T <= 623.15) {
            // Si on est en dessous de 350°C (623.15K)
            // On compare P à la pression de saturation à cette température
            double Psat = saturation_pressure(T);

            // le liquide -> Région 1
            if (P > Psat) return r1;

            if (P == Psat) return r4;

            // vapeur -> Région 2
            return r2;
        }

        // Si on est au dessus de 350°C (623.15K)
        // frontière entre vapeur et zone critique
        double P_boundary = boundary_2_3_pressure(T);

        if (P > P_boundary) {
            // Au dessus de la frontière -> Région 3
            return r3;
        }

        // En dessous -> Région 2
        return r2;
    }
};

#endif //NNPWS_REGIONS_HXX