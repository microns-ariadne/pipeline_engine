#include <math.h>
class vector3D {
public:
  float x;
  float y;
  float z;
  vector3D();
  vector3D(float _x, float _y, float _z);
  vector3D(vector3D* b);

  vector3D* multiply(float q);
  char* toString();
  float innerProduct(vector3D *b);
  void add(vector3D *b);
  void normalize();
};

vector3D::vector3D() {
  x=0;
  y=0;
  z=0;
}

vector3D::vector3D(float _x, float _y, float _z) {
  x=_x;
  y=_y;	
  z=_z;
}

vector3D::vector3D(vector3D* b) {
  x=b->x;     
  y=b->y;       
  z=b->z;
}

vector3D* vector3D::multiply(float q) {
   vector3D* temp = new vector3D(x*q, y*q, z*q);
   return temp;
}

void vector3D::add(vector3D *b) {
  x+=b->x;     
  y+=b->y;       
  z+=b->z;
}

char* vector3D::toString() {
   char* temp = new char[50];
   sprintf(temp,"(%f, %f, %f)", x, y, z);
   return temp;
}

float vector3D::innerProduct(vector3D *b) {
   return x*b->x+y*b->y+z*b->z; 
}

void vector3D::normalize() {
  float size = sqrt(this->innerProduct(this));
  if (size>0){ 
    x/=size;
    y/=size;
    z/=size;
  }
}
