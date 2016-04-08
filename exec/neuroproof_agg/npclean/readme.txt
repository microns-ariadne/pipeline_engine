Step-by-step process to build this:
NOTE: add libraries versions here


0. Have gcc with cilk (ie. gcc 4.8 compiled with cilk, or ubuntu's gcc 5.0?)
1. Install boost
2. Install hdf5.
   On Ubuntu: apt-get install libhdf5-dev
3. Install vigra (http://hci.iwr.uni-heidelberg.de/vigra) - used for h5 files
   don't use apt-get (doesn't include support for hdf5 for whatever reason)
   need to build from source
   		      DO NOT DO THIS: On Ubuntu: apt-get install libvigraimpex-dev
   do this manually instead????
4. Install JsonCpp
   apt-get install libjsoncpp-dev
5. install python-dev
   (apt-get install python-dev)
   NOTE: figure out if we really need this for our purpose
6. Install opencv
7. Install vtk
   (apt-get install libvtk5-dev)


Other:
Maybe install Gala for future use?

Display gcc's include path:
echo | cpp -Wp,-v 
/usr/local/lib/libvigraimpex.so
