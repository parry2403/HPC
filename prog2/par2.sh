#PBS -q class
#PBS -l nodes=4:sixcore
#PBS -l walltime=02:00:00
#PBS -N 10_34_8

export PBS_O_WORKDIR=$HOME/prog2

INPUT_FILE_A=$PBS_O_WORKDIR/input_A.bin
INPUT_FILE_B=$PBS_O_WORKDIR/input_b.bin
OUTPUT_FILE=$PBS_O_WORKDIR/out_x.bin
EXE=$PBS_O_WORKDIR/jacobi
p=1
n=3000
d=.8



for n in 1000 2000 4000 6000 8000 10000
do
echo "----------Parallel--n-fixed---------" >&2
echo "n=$n" >&2
for p in 1 4 9 16 25 36 49
do
echo "----------Parallel--p-fixed---------" >&2
echo "p=$p" >&2
for d in .2 .4 .5 .6 .8 1.0
do
echo "----------Parallel--d-varying---------" >&2
echo "d=$d" >&2
OMPI_MCA_mpi_yield_when_idle=0 mpirun --hostfile $PBS_NODEFILE -np $p $EXE  -n $n -d $d
done
done
done