#include "realtimeRenderf.hpp"

void cameraMove::left(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed) {
    Vector3D r = unitVec(Cross(direction, up));
    eye += speed * 0.1 * r;
}

void cameraMove::right(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed) {
    Vector3D r = unitVec(Cross(direction, up));
    eye -= speed * 0.1 * r;
}

void cameraMove::front(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed) {
    eye += speed * 0.1 * unitVec(direction);
}

void cameraMove::back(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed) {
    eye -= speed * 0.1 * unitVec(direction);
}

void cameraMove::up(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed) {
    eye += speed * 0.1 * unitVec(up);
}

void cameraMove::down(Vector3D& eye, Vector3D& direction, Vector3D& up, double speed) {
    eye -= speed * 0.1 * unitVec(up);
}

void cameraMove::rotateDirection(const sf::Vector2i& diff, Vector3D& direction, Vector3D& up, int width, int height, double speed) {
    Vector3D unitDir = unitVec(direction);
    Vector3D unitUp = unitVec(up - Dot(up, unitDir) * unitDir);
    Vector3D unitRight = Cross(unitUp, unitDir);
    Vector3D change = diff.x * unitRight + diff.y * unitUp;
    double xangle = speed*double(diff.x) / width, yangle = speed*double(diff.y) / height;
    Matrix rotationx = generateRotation(xangle, up);
    direction = rotationx * direction; up = rotationx * up;
    Matrix rotationy = generateRotation(yangle, unitRight);
    direction = unitVec(rotationy * direction); up = rotationy * up;
    up = unitVec(up - Dot(up,direction)*direction);
}

void cameraMove::roll(const Vector3D& direction, Vector3D& up, double xangle) {
    Matrix rotationx = generateRotation(xangle, direction);
    up = rotationx * up;
}

void realtimeRender::centerMouse(sf::Vector2i center, sf::Window& window) {
    sf::Mouse::setPosition(center, window);
}

void realtimeRender::detectKey(const sf::Event& event, Vector3D& eye, Vector3D& direction, Vector3D& up, bool& changes, int& speed) {
    static double speedList[2] = {1.0, 20.0};
    speed = speed % 2;
    switch (event.key.code) {
        case sf::Keyboard::Left:
            // Left
            cameraMove::left(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::A:
            // Left
            cameraMove::left(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::Right:
            // Right
            cameraMove::right(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::D:
            // Right
            cameraMove::right(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::Down:
            // Backwards
            cameraMove::back(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::S:
            // Backwards
            cameraMove::back(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::Up:
            // Forwards
            cameraMove::front(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::W:
            // Forwards
            cameraMove::front(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::Space:
            // Up
            cameraMove::up(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::LShift:
            // Down
            cameraMove::down(eye, direction, up, speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::Q:
            // Negative roll
            cameraMove::roll(direction, up, 0.01 * speedList[speed]);
            changes = true;
            break;
        case sf::Keyboard::E:
            // Positive roll
            cameraMove::roll(direction, up, -0.01 * speedList[speed]);
            changes = true;
            break;
    }
}
