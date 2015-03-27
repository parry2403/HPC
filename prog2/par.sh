#PBS -q class
#PBS -l nodes=4:sixcore
#PBS -l walltime=02:00:00
#PBS -N test

export PBS_O_WORKDIR=$HOME/prog2

EXE=$PBS_O_WORKDIR/jacobi
p=1
n=3000
d=.8



for n in 1000 
do
echo "----------Parallel--n-fixed---------" >&2
echo "n=$n" >&2
for p in 1 4 
do
echo "----------Parallel--p-fixed---------" >&2
echo "p=$p" >&2
for d in .2 
do
echo "----------Parallel--d-varying---------" >&2
echo "d=$d" >&2

OMPI_MCA_mpi_yield_when_idle=0 mpirun --hostfile $PBS_NODEFILE -np 9 $EXE -n 2000 -d .3
done
done
done