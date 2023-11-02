//This code does the following :
//1. Rotate a given vector using quaternions
//2. Converting rotation quaternion to rotation matrix
//3. Do the above operations for 1000 random points, axis and angles(line 131 needs to be uncommented while line 128 commented).

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <cstdlib>  // for random values

using namespace std;

// Define Quaternions structure 
class Quaternion {
public:
    double w, x, y, z;
    // Define Quaternion Point
    Quaternion(double w = 0, double x = 0, double y = 0, double z = 0)
        : w(w), x(x), y(y), z(z) {}

    // Define Quaternion Conjugate structure
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }
    // Define Quaternion * Conjugate operation
    Quaternion operator*(const Quaternion& q2) const {
        return Quaternion(
            w * q2.w - x * q2.x - y * q2.y - z * q2.z,
            w * q2.x + x * q2.w + y * q2.z - z * q2.y,
            w * q2.y - x * q2.z + y * q2.w + z * q2.x,
            w * q2.z + x * q2.y - y * q2.x + z * q2.w
        );
    }
};

// Define 3D-point structure
class Point3D {
public:
    double x, y, z;

    Point3D(double x = 0, double y = 0, double z = 0)
        : x(x), y(y), z(z) {}

    Quaternion toQuaternion() const {
        Quaternion q(0, x, y, z);
        return q;
    }
};

// Define the Rotation Quaternion parameters
Quaternion getRotationQuaternion(double angle, const Point3D & axis) {
    double theta = angle * M_PI / 180.0; // convert to radians
    // Calculate axis magnitude
    double magnitude = sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    // Normalize the rotation axis (sum up to 1)
    double nx = axis.x / magnitude;
    double ny = axis.y / magnitude;
    double nz = axis.z / magnitude;
    // Calculate the Rotation Quaternion
    double qw = cos(theta / 2.0);
    double qx = nx * sin(theta / 2.0);
    double qy = ny * sin(theta / 2.0);
    double qz = nz * sin(theta / 2.0);

    // create Quaternion object
    Quaternion q(qw, qx, qy, qz);
    return q;
}

// Generate random data for points, vectors, and angles
void generateRandomData(vector<Point3D>&points, vector<Point3D>&vectors, vector<double>&angles, int count) {
    for (int i = 0; i < count; i++) {
        double x_point = (rand() % 1000 - 500) / 10.0;  // Random values between -50 and 50
        double y_point = (rand() % 1000 - 500) / 10.0;
        double z_point = (rand() % 1000 - 500) / 10.0;

        double x_vector = (rand() % 1000 - 500) / 10.0;  // Random values between -50 and 50
        double y_vector = (rand() % 1000 - 500) / 10.0;
        double z_vector = (rand() % 1000 - 500) / 10.0;

        // Instances - create random xyz point values & vector + angle value
        points.push_back(Point3D(x_point, y_point, z_point));
        vectors.push_back(Point3D(x_vector, y_vector, z_vector));
        angles.push_back(rand() % 360);  // Random values between 0 and 359
    }
}

// Set data to original single-element parameters
void originalData(vector<Point3D>&points, vector<Point3D>&vectors, vector<double>&angles) {
    points.push_back(Point3D(-5, 2, 1.0 / 3));  // point xyz
    vectors.push_back(Point3D(4, -2, 3));       // rotation axis/vector
    angles.push_back(25);                       // rotation angle
}

// Define 3x3 Matrix for Quaternion to MatrixRotation convertion
class Matrix3x3 {
public:
    double m[3][3];

    // Initialize from a quaternion to 3x3 Matrix
    Matrix3x3(const Quaternion& q) {
        m[0][0] = 1 - 2 * (q.y * q.y + q.z * q.z);
        m[0][1] = 2 * (q.x * q.y - q.w * q.z);
        m[0][2] = 2 * (q.x * q.z + q.w * q.y);

        m[1][0] = 2 * (q.x * q.y + q.w * q.z);
        m[1][1] = 1 - 2 * (q.x * q.x + q.z * q.z);
        m[1][2] = 2 * (q.y * q.z - q.w * q.x);

        m[2][0] = 2 * (q.x * q.z - q.w * q.y);
        m[2][1] = 2 * (q.y * q.z + q.w * q.x);
        m[2][2] = 1 - 2 * (q.x * q.x + q.y * q.y);
    }

    // Print out the Rotation Matrix
    void print() const {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cout << m[i][j] << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    // Generate random data
    vector<Point3D> points, rotationAxes;
    vector<double> angles;

    // For original point values:
    originalData(points, rotationAxes, angles); // uncomment to see + comment (*1) below

    // For 1000 random pairs of point>axis>angleRotation
    //generateRandomData(points, rotationAxes, angles, 100); //(*1)

    for (int i = 0; i < points.size(); i++) {
        Point3D& point = points[i];
        Point3D& rotationAxis = rotationAxes[i];
        double& angle = angles[i];

        // Debug: Print original point
        cout << "Original Point " << i + 1 << ": [" << point.x << ", " << point.y << ", " << point.z << "]" << endl;

        // Calculate point to quaternion, rotation quaternion, rotated quaternion 
        Quaternion pointQuaternion = point.toQuaternion();
        Quaternion rotationQuaternion = getRotationQuaternion(angle, rotationAxis);
        Quaternion rotatedQuaternion = rotationQuaternion * pointQuaternion * rotationQuaternion.conjugate();

        // Debug: Print rotated point values
        cout << "Rotated Point with Quaternion - point#" << i + 1 << ": \n[" << rotatedQuaternion.x << ", " << rotatedQuaternion.y << ", " << rotatedQuaternion.z << "]" << endl;

        // Convertion function for Quaternion to Matrix Rotation
        Matrix3x3 rotationMatrix(rotatedQuaternion);
        cout << "Rotation Quaternion to Rotation Matrix is:\n[" << endl;
        rotationMatrix.print();
        cout << "]" << endl;
    }
    return 0;
}