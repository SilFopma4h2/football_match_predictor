from torch_predictor import use_winner
from ml_predictor import get_winner
from ml_predictor_plot import plot_tree_ml


print(r"""
     /$$$$$$$$                    /$$     /$$                 /$$ /$$                                           /$$ /$$             /$$                        
    | $$_____/                   | $$    | $$                | $$| $$                                          | $$|__/            | $$                        
    | $$     /$$$$$$   /$$$$$$  /$$$$$$  | $$$$$$$   /$$$$$$ | $$| $$        /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$$ /$$  /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$ 
    | $$$$$ /$$__  $$ /$$__  $$|_  $$_/  | $$__  $$ |____  $$| $$| $$       /$$__  $$ /$$__  $$ /$$__  $$ /$$__  $$| $$ /$$_____/|_  $$_/   /$$__  $$ /$$__  $$
    | $$__/| $$  \ $$| $$  \ $$  | $$    | $$  \ $$  /$$$$$$$| $$| $$      | $$  \ $$| $$  \__/| $$$$$$$$| $$  | $$| $$| $$        | $$    | $$  \ $$| $$  \__/
    | $$   | $$  | $$| $$  | $$  | $$ /$$| $$  | $$ /$$__  $$| $$| $$      | $$  | $$| $$      | $$_____/| $$  | $$| $$| $$        | $$ /$$| $$  | $$| $$      
    | $$   |  $$$$$$/|  $$$$$$/  |  $$$$/| $$$$$$$/|  $$$$$$$| $$| $$      | $$$$$$$/| $$      |  $$$$$$$|  $$$$$$$| $$|  $$$$$$$  |  $$$$/|  $$$$$$/| $$      
    |__/    \______/  \______/    \___/  |_______/  \_______/|__/|__/      | $$____/ |__/       \_______/ \_______/|__/ \_______/   \___/   \______/ |__/      
                                                                           | $$                                                                                
                                                                           | $$                                                                                
                                                                           |__/                                                                                
""")

print("Welcome by the best football predictor ever!!")
print("Type help to get started.")
print("Type 'ml' for the ML model to predict the winner.")
print("Type 'torch' for the torch model to predict the winner.")
print("Type 'both' to compare the two models.")
print("Type 'ml_plot' to plot the tree.")
print("Type exit to exit the program.")

while True:
    user_input = input("Enter your prediction model: ")

    if user_input == "ml":
        print("The winner is: ", get_winner())

    elif user_input == "torch":
        print("The winner is: ", use_winner())

    elif user_input == "both":
        res_te = use_winner()
        res_ml = get_winner()

        if res_te == res_ml:
            print("The predicted winner from ml is same as the winner from torch.")

        elif res_te != res_ml:
            print("The predicted winner from ml is different from the winner from torch.")

    elif user_input == "help":
        print("Type 'ml' for the ML model to predict the winner.")
        print("Type 'torch' for the torch model to predict the winner.")
        print("Type 'both' to compare the two models.")


    elif user_input == "exit":
        print("Goodbye!")
        break

    elif user_input == "ml_plot":
        plot_tree_ml()

    else:
        print("Invalid input. Please try again.")