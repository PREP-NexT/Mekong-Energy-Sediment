N=10

(
for l in 1 2 4 6; do
	for c in 1 2 3 4; do
		for s in $(seq 14 0.2 54); do
			((i=i%N)); ((i++==0)) && wait
			python run_mekong_gurobi.py --carbon=$c --sediment=$s --limit=$l &
		done
	done
done
wait
)
