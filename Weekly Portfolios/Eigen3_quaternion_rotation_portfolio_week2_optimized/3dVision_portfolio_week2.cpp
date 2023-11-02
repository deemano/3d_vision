#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <chrono>  // for timer

using namespace std;
using namespace Eigen;

// 1. Represent the Point as a Quaternion
Quaterniond pointToQuaternion(const Vector3d& point) {
    return Quaterniond(0, point[0], point[1], point[2]);
}

// 2. Normalize the Axis of Rotation
Vector3d normalizeAxis(const Vector3d& axis) {
    return axis.normalized();
}

// 3. Compute the Rotation Quaternion
Quaterniond computeRotationQuaternion(double angle, const Vector3d& axis) {
    AngleAxisd rotation(angle * M_PI / 180.0, axis);
    return Quaterniond(rotation);
}

// 4. Compute the Conjugate of the Rotation Quaternion
Quaterniond conjugateQuaternion(const Quaterniond& q) {
    return q.conjugate();
}

// 5. Rotate the Point
Quaterniond rotatePoint(const Quaterniond& pointQuaternion, const Quaterniond& rotationQuaternion) {
    Quaterniond rotationConjugate = conjugateQuaternion(rotationQuaternion);
    return rotationQuaternion * pointQuaternion * rotationConjugate;
}

// 6. Extract the Rotated Point's Coordinates
Vector3d extractRotatedPointCoordinates(const Quaterniond& rotatedPointQuaternion) {
    return Vector3d(rotatedPointQuaternion.x(), rotatedPointQuaternion.y(), rotatedPointQuaternion.z());
}

// Original point - 25 degress quaternion rotation around axis
void originalRot(vector<Vector3d>& points, vector<Vector3d>& axes, vector<double>& angles) {
    points.resize(1); // seting all to have 1 element
    axes.resize(1);
    angles.resize(1);

    points[0] = Vector3d(-5, 2, 1.0 / 3);   // Point to be rotated
    axes[0] = Vector3d(4, -2, 3);           // Rotation axis
    angles[0] = 25;                         // Rotation angle in degrees
}

// Generate a # of random operations & results
void generateRandomData(vector<Vector3d>& points, vector<Vector3d>& axes, vector<double>& angles, int num_samples = 1000) {
    points.resize(num_samples);
    axes.resize(num_samples);
    angles.resize(num_samples);

    for (int i = 0; i < num_samples; i++) {
        // Generating random point values between -10 and 10
        points[i] = Vector3d::Random() * 10;

        // Generating random axis values
        axes[i] = Vector3d::Random();

        // Generating random angle between 0 and 360 degrees
        angles[i] = ((double)rand() / RAND_MAX) * 360;
    }
}
// Computes rotation matrix from axis and angle
Matrix3d computeRotationMatrix(double angle, const Vector3d& axis) {
    AngleAxisd rotation(angle * M_PI / 180.0, axis);
    return rotation.toRotationMatrix();
}

// Rotates a point using the rotation matrix
Vector3d rotatePointUsingMatrix(const Vector3d& point, const Matrix3d& rotationMatrix) {
    return rotationMatrix * point;
}

int main() {
    // Define variable & objects
    vector<Vector3d> points, axes;
    vector<double> angles;

    //originalRot(points, axes, angles); // uncomment to show original single result
    generateRandomData(points, axes, angles);

    // Start benchmark timer here
    auto start_time = chrono::high_resolution_clock::now();

    // Loop to get 1000 random results
    for (int i = 0; i < points.size(); i++) {
        Quaterniond pointQuaternion = pointToQuaternion(points[i]);
        Vector3d normalizedAxis = normalizeAxis(axes[i]);
        Quaterniond rotationQuaternion = computeRotationQuaternion(angles[i], normalizedAxis);
        Quaterniond rotatedPointQuaternion = rotatePoint(pointQuaternion, rotationQuaternion);
        Vector3d rotatedPoint = extractRotatedPointCoordinates(rotatedPointQuaternion);

        // For debug only: Print results
        //cout << "Original Point: [" << points[i].transpose() << "]" << endl;
        //cout << "Rotated Point: [" << rotatedPoint.transpose() << "]" << endl;
    }
    // Stop timing here
    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    cout << "Quaternions time taken is   " << elapsed_time << " milliseconds." << endl;

    // Start benchmark timer for matrix rotation
    start_time = chrono::high_resolution_clock::now();

    // Loop to get 1000 random results using rotation matrices
    for (int i = 0; i < points.size(); i++) {
        Vector3d normalizedAxis = normalizeAxis(axes[i]);
        Matrix3d rotationMatrix = computeRotationMatrix(angles[i], normalizedAxis);
        Vector3d rotatedPoint = rotatePointUsingMatrix(points[i], rotationMatrix);
    }

    // Stop timing here for matrix rotation
    end_time = chrono::high_resolution_clock::now();
    elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "MatrixRotation time taken: " << elapsed_time << " milliseconds." << endl;

    return 0;
}