SetFactory("OpenCASCADE");

// All dimensions are millimetres.
LengthX = 470;
LengthY = 330;
LengthZ = 220;
PatchWidth = 100;
PatchHeight = 80;

// Preserve a mesh size supplied with Gmsh's -setnumber option.
If (!Exists(TargetSize))
  TargetSize = 14;
EndIf

// Wall: 0 = -X, 1 = -Y, 2 = -Z. The guards preserve values supplied with
// Gmsh's -setnumber option.
If (!Exists(PatchWall))
  PatchWall = 1;
EndIf
// Patch-centre offsets along the wall-local horizontal and vertical axes.
If (!Exists(PatchU))
  PatchU = 0;
EndIf
If (!Exists(PatchV))
  PatchV = 0;
EndIf

XMin = -LengthX / 2;
YMin = -LengthY / 2;
ZMin = 0;

volume = newv;
Box(volume) = {XMin, YMin, ZMin, LengthX, LengthY, LengthZ};

If (PatchWall == 0)
  X0 = XMin;
  Y0 = PatchU - PatchWidth / 2;
  Z0 = LengthZ / 2 + PatchV - PatchHeight / 2;
  p1 = newp; Point(p1) = {X0, Y0, Z0, TargetSize};
  p2 = newp; Point(p2) = {X0, Y0 + PatchWidth, Z0, TargetSize};
  p3 = newp; Point(p3) = {X0, Y0 + PatchWidth, Z0 + PatchHeight, TargetSize};
  p4 = newp; Point(p4) = {X0, Y0, Z0 + PatchHeight, TargetSize};
ElseIf (PatchWall == 1)
  X0 = PatchU - PatchWidth / 2;
  Y0 = YMin;
  Z0 = LengthZ / 2 + PatchV - PatchHeight / 2;
  p1 = newp; Point(p1) = {X0, Y0, Z0, TargetSize};
  p2 = newp; Point(p2) = {X0 + PatchWidth, Y0, Z0, TargetSize};
  p3 = newp; Point(p3) = {X0 + PatchWidth, Y0, Z0 + PatchHeight, TargetSize};
  p4 = newp; Point(p4) = {X0, Y0, Z0 + PatchHeight, TargetSize};
ElseIf (PatchWall == 2)
  X0 = PatchU - PatchWidth / 2;
  Y0 = PatchV - PatchHeight / 2;
  Z0 = ZMin;
  p1 = newp; Point(p1) = {X0, Y0, Z0, TargetSize};
  p2 = newp; Point(p2) = {X0 + PatchWidth, Y0, Z0, TargetSize};
  p3 = newp; Point(p3) = {X0 + PatchWidth, Y0 + PatchHeight, Z0, TargetSize};
  p4 = newp; Point(p4) = {X0, Y0 + PatchHeight, Z0, TargetSize};
Else
  Error("PatchWall must be 0 (-X), 1 (-Y), or 2 (-Z).");
EndIf

l1 = newl; Line(l1) = {p1, p2};
l2 = newl; Line(l2) = {p2, p3};
l3 = newl; Line(l3) = {p3, p4};
l4 = newl; Line(l4) = {p4, p1};
loop = newll; Curve Loop(loop) = {l1, l2, l3, l4};
patch = news; Plane Surface(patch) = {loop};

BooleanFragments{ Volume{volume}; Delete; }{ Surface{patch}; Delete; }

volumes[] = Volume{:};
allWalls[] = Boundary{ Volume{volumes[]}; };
epsilon = 1e-4;
If (PatchWall == 0)
  radiator[] = Surface In BoundingBox{
    X0 - epsilon, Y0 - epsilon, Z0 - epsilon,
    X0 + epsilon, Y0 + PatchWidth + epsilon, Z0 + PatchHeight + epsilon
  };
ElseIf (PatchWall == 1)
  radiator[] = Surface In BoundingBox{
    X0 - epsilon, Y0 - epsilon, Z0 - epsilon,
    X0 + PatchWidth + epsilon, Y0 + epsilon, Z0 + PatchHeight + epsilon
  };
Else
  radiator[] = Surface In BoundingBox{
    X0 - epsilon, Y0 - epsilon, Z0 - epsilon,
    X0 + PatchWidth + epsilon, Y0 + PatchHeight + epsilon, Z0 + epsilon
  };
EndIf
allWalls[] -= radiator[];

Physical Volume("Body1", 1) = {volumes[]};
Physical Surface("Radiator", 2) = {radiator[]};
Physical Surface("Body1_boundary", 3) = {allWalls[]};

Mesh.MeshSizeMin = TargetSize;
Mesh.MeshSizeMax = TargetSize;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm3D = 1;
Mesh.MshFileVersion = 4.1;
