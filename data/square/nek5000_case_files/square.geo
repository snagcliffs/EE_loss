// Description of domain
xi = 12;
xo = 30;
h = 12;
r = 0.5;
R = h/2;

// Grid sizes
n_inlet = 9;
n_top = 9;
n_back = 9;
n_r = 25;
n_wake = 13;

// Progression ratios
p_sf = 1/1.1; // progression along front of cylinder
p_st = 1/1.05; // progression along top of cylinder
p_sb = 1/1.05; // progression along back of cylinder

p_r = 1.175;  // progression along axial direction
p_R = 1.05;  // progression along y-axis in wake
p_w = 1.02;  // progression along wake in x direction

theta = Atan(h / xi);
Mesh.Smoothing=0;

// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Points

Point(0) = {0,0,0,0};

Point(1) = {-r,0,0,0};
Point(2) = {-r,r,0,0};
Point(3) = {0,r,0,0};
Point(4) = {r,r,0,0};
Point(5) = {r,0,0,0};
Point(6) = {r,-r,0,0};
Point(7) = {0,-r,0,0};
Point(8) = {-r,-r,0,0};


Point(9) = {-xi,0,0,0};
Point(10) = {-xi,h,0,0};
Point(11) = {0,h,0,0};
Point(12) = {xi,h,0,0};
Point(13) = {xi,0,0,0};
Point(14) = {xi,-h,0,0};
Point(15) = {0,-h,0,0};
Point(16) = {-xi,-h,0,0};

Point(17) = {xo,h,0,0};
Point(18) = {xo,0,0,0};
Point(19) = {xo,-h,0,0};

// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Lines

Line(1) = {1,2}; Transfinite Curve {1} = n_inlet Using Progression p_sf;
Line(2) = {1,8}; Transfinite Curve {2} = n_inlet Using Progression p_sf;
Line(3) = {2,3}; Transfinite Curve {3} = n_top Using Progression 1/p_st;
Line(4) = {8,7}; Transfinite Curve {4} = n_top Using Progression 1/p_st;
Line(5) = {3,4}; Transfinite Curve {5} = n_top Using Progression p_st;
Line(6) = {7,6}; Transfinite Curve {6} = n_top Using Progression p_st;
Line(7) = {5,4}; Transfinite Curve {7} = n_back Using Progression p_sb;
Line(8) = {5,6}; Transfinite Curve {8} = n_back Using Progression p_sb;

Line(9) = {9,10}; Transfinite Curve {9} = n_inlet;
Line(10) = {9,16}; Transfinite Curve {10} = n_inlet;
Line(11) = {10,11}; Transfinite Curve {11} = n_top;
Line(12) = {16,15}; Transfinite Curve {12} = n_top;
Line(13) = {11,12}; Transfinite Curve {13} = n_top;
Line(14) = {15,14}; Transfinite Curve {14} = n_top;
Line(15) = {13,12}; Transfinite Curve {15} = n_back Using Progression p_R;
Line(16) = {13,14}; Transfinite Curve {16} = n_back Using Progression p_R;


Line(17) = {1,9}; Transfinite Curve {17} = n_r Using Progression p_r;
Line(18) = {2,10}; Transfinite Curve {18} = n_r Using Progression p_r;
Line(19) = {3,11}; Transfinite Curve {19} = n_r Using Progression p_r;
Line(20) = {4,12}; Transfinite Curve {20} = n_r Using Progression p_r;
Line(21) = {5,13}; Transfinite Curve {21} = n_r Using Progression p_r;
Line(22) = {6,14}; Transfinite Curve {22} = n_r Using Progression p_r;
Line(23) = {7,15}; Transfinite Curve {23} = n_r Using Progression p_r;
Line(24) = {8,16}; Transfinite Curve {24} = n_r Using Progression p_r;

Line(25) = {12,17}; Transfinite Curve {25} = n_wake Using Progression p_w;
Line(26) = {13,18}; Transfinite Curve {26} = n_wake Using Progression p_w;
Line(27) = {14,19}; Transfinite Curve {27} = n_wake Using Progression p_w;

Line(28) = {18,17}; Transfinite Curve {28} = n_back Using Progression p_R;
Line(29) = {18,19}; Transfinite Curve {29} = n_back Using Progression p_R;

// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Surfaces

Line Loop(1) = {1,18,-9,-17};  Plane Surface(1) = {1};
Line Loop(2) = {3,19,-11,-18};  Plane Surface(2) = {2};
Line Loop(3) = {5,20,-13,-19};  Plane Surface(3) = {3};
Line Loop(4) = {-7,21,15,-20};  Plane Surface(4) = {4};
Line Loop(5) = {8,22,-16,-21};  Plane Surface(5) = {5};
Line Loop(6) = {-6,23,14,-22};  Plane Surface(6) = {6};
Line Loop(7) = {-4,24,12,-23};  Plane Surface(7) = {7};
Line Loop(8) = {-2,17,10,-24};  Plane Surface(8) = {8};
Line Loop(9) = {15,25,-28,-26};  Plane Surface(9) = {9};
Line Loop(10) = {-16,26,29,-27};  Plane Surface(10) = {10};

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {7};
Transfinite Surface {8};
Transfinite Surface {9};
Transfinite Surface {10};

Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};
Recombine Surface {4};
Recombine Surface {5};
Recombine Surface {6};
Recombine Surface {7};
Recombine Surface {8};
Recombine Surface {9};
Recombine Surface {10};

Physical Line("Inlet") = {9,10};
Physical Line("Outlet") = {28,29};
Physical Line("Symm") = {11,12,13,14,25,27};
Physical Line("Wall") = {1,2,3,4,5,6,7,8};
Physical Surface("Fluid") = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

