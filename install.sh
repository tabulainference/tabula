# Install yao gc
gitrootdir=`git rev-parse --show-toplevel`

case "$(uname -s)" in

   Darwin)
       echo 'Mac OS X'

       pushd $gitrootdir/src/MP-SPDZ-yao-interactive
       rm -rf $gitrootdir/src/MP-SPDZ-yao-interactive/local/lib
       make clean
       make -j 8 yao
       mkdir -p releases/mac/
       cp -f yao-party.x releases/mac/yao-party-interactive.x
       popd
       
       cp $gitrootdir/src/MP-SPDZ/releases/mac/yao-party.x $gitrootdir/src/garbled_circuits/
       cp $gitrootdir/src/MP-SPDZ-yao-interactive/releases/mac/yao-party-interactive.x $gitrootdir/src/garbled_circuits/              
       ;;
   Linux)
       echo 'Linux'

       pushd $gitrootdir/src/MP-SPDZ-yao-interactive
       export LD_LIBRARY_PATH=$gitrootdir/src/MP-SPDZ-yao-interactive/local/lib
       make clean
       make -j 8 yao
       mkdir -p releases/linux/
       cp yao-party.x releases/linux/yao-party-interactive.x
       popd
       
       cp $gitrootdir/src/MP-SPDZ/releases/linux/yao-party.x $gitrootdir/src/garbled_circuits/
       cp $gitrootdir/src/MP-SPDZ-yao-interactive/releases/linux/yao-party-interactive.x $gitrootdir/src/garbled_circuits/                     
       ;;
     
   CYGWIN*|MINGW32*|MSYS*|MINGW*)
     echo 'MS Windows'
     ;;

   # Add here more strings to compare
   # See correspondence table at the bottom of this answer

   *)
     echo 'Other OS' 
     ;;
esac


cd $gitrootdir
pushd src/fast_hash
#python setup.py build_ext --inplace
popd
