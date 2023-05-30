from RemovingEdgesPCStable import *

edges = [()]

if len(sys.argv) != 2:
    print("USAGE: RemovingEdgesPCStable [train_file.csv]")
    print("EXAMPLE1: RemovingEdgesPCStable play_tennis-train-x3.csv")
    print("EXAMPLE2: RemovingEdgesPCStable play_tennis-train-x3.csv")
    print("EXAMPLE3: RemovingEdgesPCStable play_tennis-train-x3.csv")
    exit(0)
else:
    print("hi")
    data_file = sys.argv[1]
    PcStable = RemovingEdgesPCStable(data_file)
    PcStable.init_structure()
    print("init")
    print("1.Conditional Independence For Random Structure\n2.PC Skeleton")
    choice = input("Enter Choice:")
    
    if choice == '1':
        PcStable.remove_edges_between_nodes()
    elif choice == '2':
        PcStable.parents_of_node()
        PcStable.add_graph_edges(choice)
        PcStable.PC_stable_skeleton()

