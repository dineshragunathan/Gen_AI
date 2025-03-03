import numpy as np
import matplotlib.pyplot as plt

'''
my understanding of the concept and what i have implemented : 

given a set of data points that are linearly dependent , we are to find the line of best fit (which
represents the data points the best using a model) using gradient descent , which is a tool used to 
find the optimal values of m1 and m2 , i.e the slope and intercept of the line of best fit.

what i have done : 

firstly i created my data set to work on by using the numpy lib , i created the x data points 
first and created the corresponding y values using a random line equation and noise . 
after creating the data sets , i get started by coding the grad_descent function.

To get the optimal values for m1 and m2 , which are the slope and intercept in our line of best fit ,
we calculate the loss for m1 = m2 = 0 initially and continually improve upon it...
(to calculate the loss we are using MSE)

In each iteration of the descent function we calculate the slope of the loss function with respect
to the parameters m1 and m2 , which gives us the slopes of the plots - #1 loss function vs m1 and 
#2 loss function vs m2 and which ultimately tells us the direction for us to head in for each
parameter. (Since we are trying to head to the bottom of the curve..) 

So after each iteration i update the values of m1 and m2. (directions are handled in the code),
and in each iteration the cost is reduced and we get better values of m1 and m2 , that best represent
the data points

higher the number of iterations , better the values of m1 and m2 of best fit line.
learning rate represents the step size used to update the values of m1 and m2 , high values 
may cause the program to skip the minima , which would result in increased number of iterations to
get back there.

I've tried different learning rates and the learning rate in the order of 1e^-3 was the sweet spot.

Atlast i've used a graph to plot the data points and show the line of best fit i've obtained after
the given set of iterations.  
'''

''' Gradient Descent Implementation '''

x = np.random.normal(15 ,5 , 50)

print(f'The x data points are : \n {x} \n')

#generating y  , for x data points
y = 2*x + np.random.normal(0,12,50)

print(f'The y data points are : \n {y} \n')

#setting up the parameters to be found
m1 = m2 = 0

#gradient descent function
def descent(x ,y, m1 , m2 , learning_rate):

    n = x.shape[0]

    #since we're using mse for calculating loss the partial derivative would be as follows
    slope_m1 = -(2/n)*sum(x*(y - (m1*x + m2)))
    slope_m2 = -(2/n)*sum(y - (m1*x + m2))

    m1 -= slope_m1*learning_rate
    m2 -= slope_m2*learning_rate

    return m1 , m2

#hyperparameters
iterations = 10000
learning_rate = 0.00100001

#lists for storing values to visualize data
loss_values = []
iteration_value = []
m1_values = []
m2_values = []

#calling the gradient_descent function on the dataset
for i in range(iterations):

    m1 , m2 = descent(x , y , m1 , m2 , learning_rate)
    N = x.shape[0]
    loss = sum(y - (m1*x + m2))**2 / N
    loss_values.append(loss)
    iteration_value.append(i)
    m1_values.append(m1)
    m2_values.append(m2)
    if i%2000 == 0:
        print(f'cost {loss} m1 {m1} m2 {m2} Iteration {i}' ,end ="\n")

print()
print(f'final m1 : {m1} final m2 : {m2} from Gradient Descent \n')


''' Linear Search Algorithm Implementation '''

#generating a random search space for possible optimal values for m1 and m2
# since I know the answers range from the grad_descent , im using a narrow search space

m1_range = np.linspace(-3,3 , 100)
m2_range = np.linspace(-2 , 2 , 100)

#initializing variable to find
opt_m1 = opt_m2 = 0
cost_linear = 0.0
iteration_count = 0
best_iteration = 0

#initializing variable for data visualization
loss_values_linear = []

#finding optimal values of m1 and m2 in the range linearly
for potential_m1 in m1_range:
    for potential_m2 in m2_range:

        #initializing the first cost value
        if cost_linear == 0.0:
            cost_linear = (1/100) * sum(y - (potential_m1*x + potential_m2))**2
            opt_m1 = potential_m1
            opt_m2 = potential_m2
            loss_values_linear.append(cost_linear)

        else:
            temp_cost = (1/100) * sum(y - (potential_m1*x + potential_m2))**2

            #comparing the previous cost values and updating m1 and m2
            if temp_cost < cost_linear:
                cost_linear = temp_cost
                opt_m1 = potential_m1
                opt_m2 = potential_m2
                best_iteration = iteration_count
            loss_values_linear.append(cost_linear)

            if iteration_count%2000 == 0:
                print(f'cost {cost_linear} m1 {potential_m1} m2 {potential_m2} Iteration {iteration_count}' , end ="\n")
        iteration_count += 1

print()
print(f'optimal m1 from linear search {opt_m1} optimal m2 {opt_m2} in Iteration {best_iteration} from linear search')


''' Data visualization '''

#data points visualization with line of best fit
plt.figure(1)
plt.title('Line of best fit')
plt.plot(x, y , 'o' ,color='orange')
y_hat = m1*x + m2
plt.plot(x , y_hat)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(["data points" , "line of best fit"] , loc="lower right")
plt.show()

#loss vs epochs graph
plt.figure(2)
plt.title('Cost vs Epochs')
plt.plot(iteration_value , loss_values , color='red')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.show()

#m1 vs loss graph
plt.figure(3)
plt.title('Cost vs m1')
plt.plot(m1_values , loss_values , color = 'green')
plt.xlabel('m1')
plt.ylabel('Cost (MSE)')
plt.show()

# #m2 vs loss graph
# plt.figure(4)
# plt.title('Cost vs m2')
# plt.plot(m2_values , loss_values , color = 'blue')
# plt.xlabel('m2')
# plt.ylabel('Cost (MSE)')
# plt.show()

# #loss vs epochs for linear
# plt.figure(5)
# plt.title('Cost vs Iterations (Linear Search)')
# plt.plot(iteration_value , loss_values_linear , color='black')
# plt.show()

plt.figure(6)
plt.title('Gradient Descent vs Linear Search')
plt.plot(iteration_value , loss_values_linear , color='red')
plt.plot(iteration_value , loss_values , color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.legend(['linear search' , 'gradient descent'])
plt.show()

'''
Linear Search Algorithm : 

Since I know the results of the line of best fit from gradient descent , The m1 and m2 range for linear
search algorithm was chosen narrow. The goal was to find the most optimal pair of values from the 
ranges by iterating through them linearly which I find out by comparing them in each iteration 
with the previous values and keep the best values , which are displayed at the end.

Observations : 

Linear search algorithm took around 8000 iterations to find the most optimal values of m1 and m2 
while gradient descent achieved it within 5 iterations. I've plotted a graph at the end 
to visualize the performance of gradient descent . Gradient descent achieved the most optimal 
values within very few iteration and the changes after that were very minimal that it appeared to 
be a horizontal line while linear search gradually reduced the cost as it neared the optimal 
values by approaching the optimal values linearly. 
'''

