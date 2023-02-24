#!/usr/bin/env python
# coding: utf-8

# # Java Programming Cheatsheet

# ## Comparison of Java and Python Programming Languages

# <table border=1>
#     <thead>
#         <tr>
#             <th>Concept</th>
#             <th>Java</th>
#             <th>Python</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>Objects and Classes</td>
#             <td>Java is a fully object-oriented language, with everything represented as an object.</td>
#             <td>Python is also object-oriented, but allows for more flexibility in terms of object-oriented programming.
#             </td>
#         </tr>
#         <tr>
#             <td>Variables and Data Types</td>
#             <td>Java is a statically-typed language, which means that variables must be declared with a specific data
#                 type.</td>
#             <td>Python is a dynamically-typed language, which means that the data type of a variable is determined at
#                 runtime.</td>
#         </tr>
#         <tr>
#             <td>Operators</td>
#             <td>Java provides a wide range of operators, including arithmetic, comparison, and logical operators.</td>
#             <td>Python also provides a range of operators, including arithmetic, comparison, and logical operators.</td>
#         </tr>
#         <tr>
#             <td>Control Structures</td>
#             <td>Java provides if-else statements, for loops, and while loops for controlling the flow of execution in a
#                 program.</td>
#             <td>Python provides if-elif-else statements, for loops, and while loops, as well as other control structures
#                 such as try-except blocks for exception handling.</td>
#         </tr>
#         <tr>
#             <td>Methods</td>
#             <td>Java allows you to define your own methods and call them from other parts of your code.</td>
#             <td>Python also allows you to define and call methods.</td>
#         </tr>
#         <tr>
#             <td>Arrays</td>
#             <td>Java provides several types of arrays, including one-dimensional and multi-dimensional arrays.</td>
#             <td>Python provides lists, which are similar to arrays but are more flexible and can store elements of
#                 different data types.</td>
#         </tr>
#         <tr>
#             <td>Exception Handling</td>
#             <td>Java provides try-catch blocks for handling exceptions.</td>
#             <td>Python provides try-except blocks for handling exceptions.</td>
#         </tr>
#         <tr>
#             <td>Memory Management</td>
#             <td>Java requires explicit memory management with features like garbage collection.</td>
#             <td>Python has automatic memory management and garbage collection.</td>
#         </tr>
#         <tr>
#             <td>Platform Independence</td>
#             <td>Java programs can run on any platform with JVM.</td>
#             <td>Python programs can run on any platform with Python interpreter.</td>
#         </tr>
#         <tr>
#             <td>Performance</td>
#             <td>Java programs are generally faster than Python programs.</td>
#             <td>Python programs are generally slower than Java programs.</td>
#         </tr>
#         <tr>
#             <td>Learning Curve</td>
#             <td>Java has a steeper learning curve than Python.</td>
#             <td>Python has a relatively easier learning curve than Java.</td>
#         </tr>
#         <tr>
#             <td>Libraries and Frameworks</td>
#             <td>Java has a vast collection of libraries and frameworks for various purposes.</td>
#             <td>Python also has a large collection of libraries and frameworks, but comparatively fewer than Java.</td>
#         </tr>
#         <tr>
#             <td>Usage</td>
#             <td>Java is commonly used for enterprise applications, Android development, and large-scale web
#                 applications.</td>
#             <td>Python is commonly used for scientific computing, data analysis, web development, artificial
#                 intelligence, and machine learning.</td>
#         </tr>
#     </tbody>
# </table>

# ## Datatypes

# <table border=1>
#     <thead>
#         <tr>
#             <th>Data Type</th>
#             <th>Description</th>
#             <th>Syntax Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>byte</td>
#             <td>A 1-byte integer</td>
#             <td><code>byte b = 127;</code></td>
#         </tr>
#         <tr>
#             <td>short</td>
#             <td>A 2-byte integer</td>
#             <td><code>short s = 32767;</code></td>
#         </tr>
#         <tr>
#             <td>int</td>
#             <td>A 4-byte integer</td>
#             <td><code>int i = 2147483647;</code></td>
#         </tr>
#         <tr>
#             <td>long</td>
#             <td>An 8-byte integer</td>
#             <td><code>long l = 9223372036854775807L;</code></td>
#         </tr>
#         <tr>
#             <td>float</td>
#             <td>A 4-byte floating point number</td>
#             <td><code>float f = 3.14159f;</code></td>
#         </tr>
#         <tr>
#             <td>double</td>
#             <td>An 8-byte floating point number</td>
#             <td><code>double d = 3.141592653589793;</code></td>
#         </tr>
#         <tr>
#             <td>boolean</td>
#             <td>A boolean value of either <code>true</code> or <code>false</code></td>
#             <td><code>boolean b = true;</code></td>
#         </tr>
#         <tr>
#             <td>char</td>
#             <td>A single Unicode character</td>
#             <td><code>char c = 'a';</code></td>
#         </tr>
#         <tr>
#             <td>String</td>
#             <td>A sequence of characters</td>
#             <td><code>String str = "Hello, world!";</code></td>
#         </tr>
#     </tbody>
# </table>

# ## Operators

# #### **Arithmetic Operators**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>+</td>
#             <td>Addition operator adds two operands.</td>
#             <td><code>int sum = 5 + 3; // sum is 8</code></td>
#         </tr>
#         <tr>
#             <td>-</td>
#             <td>Subtraction operator subtracts two operands.</td>
#             <td><code>int difference = 5 - 3; // difference is 2</code></td>
#         </tr>
#         <tr>
#             <td>*</td>
#             <td>Multiplication operator multiplies two operands.</td>
#             <td><code>int product = 5 * 3; // product is 15</code></td>
#         </tr>
#         <tr>
#             <td>/</td>
#             <td>Division operator divides two operands.</td>
#             <td><code>int quotient = 5 / 3; // quotient is 1</code></td>
#         </tr>
#         <tr>
#             <td>%</td>
#             <td>Modulus operator returns the remainder after division of two operands.</td>
#             <td><code>int remainder = 5 % 3; // remainder is 2</code></td>
#         </tr>
#     </tbody>
# </table>

# #### **Assignment Operators**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>=</td>
#             <td>Simple assignment operator assigns a value to a variable.</td>
#             <td><code>int x = 5; // x is 5</code></td>
#         </tr>
#         <tr>
#             <td>+=</td>
#             <td>Add and assign operator adds the right operand to the left operand and assigns the result to the left
#                 operand.</td>
#             <td><code>int x = 5; x += 3; // x is 8</code></td>
#         </tr>
#         <tr>
#             <td>-=</td>
#             <td>Subtract and assign operator subtracts the right operand from the left operand and assigns the result to
#                 the left operand.</td>
#             <td><code>int x = 5; x -= 3; // x is 2</code></td>
#         </tr>
#         <tr>
#             <td>*=</td>
#             <td>Multiply and assign operator multiplies the right operand with the left operand and assigns the result
#                 to the left operand.</td>
#             <td><code>int x = 5; x *= 3; // x is 15</code></td>
#         </tr>
#         <tr>
#             <td>/=</td>
#             <td>Divide and assign operator divides the left operand by the right operand and assigns the result to the
#                 left operand.</td>
#             <td><code>int x = 5; x /= 3; // x is 1</code></td>
#         </tr>
#         <tr>
#             <td>%=</td>
#             <td>Modulus and assign operator returns the remainder after division of the left operand by the right
#                 operand and assigns the result to the left operand.</td>
#             <td><code>int x = 5; x %= 3; // x is 2</code></td>
#         </tr>
#     </tbody>
# </table>

# #### **Comparison Operators**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>==</td>
#             <td>Equal to operator compares two operands for equality.</td>
#             <td><code>boolean isEqual = 5 == 3; // isEqual is false</code></td>
#         </tr>
#         <tr>
#             <td>!=</td>
#             <td>Not equal to operator compares two operands for inequality.</td>
#             <td><code>boolean isNotEqual = 5 != 3; // isNotEqual is true</code></td>
#         </tr>
#         <tr>
#             <td>&gt;</td>
#             <td>Greater than operator compares two operands and returns true if the left operand is greater than the
#                 right operand.</td>
#             <td><code>boolean isGreaterThan = 5 &gt; 3; // isGreaterThan is true</code></td>
#         </tr>
#         <tr>
#             <td>&lt;</td>
#             <td>Less than operator compares two operands and returns true if the left operand is less than the right
#                 operand.</td>
#             <td><code>boolean isLessThan = 5 &lt; 3; // isLessThan is false</code></td>
#         </tr>
#         <tr>
#             <td>&gt;=</td>
#             <td>Greater than or equal to operator compares two operands and returns true if the left operand is greater
#                 than or equal to the right operand.</td>
#             <td><code>boolean isGreaterThanOrEqual = 5 &gt;= 3; // isGreaterThanOrEqual is true</code></td>
#         </tr>
#         <tr>
#             <td>&lt;=</td>
#             <td>Less than or equal to operator compares two operands and returns true if the left operand is less than
#                 or equal to the right operand.</td>
#             <td><code>boolean isLessThanOrEqual = 5 &lt;= 3; // isLessThanOrEqual is false</code></td>
#         </tr>
#     </tbody>
# </table>

# #### **Logical Operators**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>&amp;&amp;</td>
#             <td>Logical AND operator returns true if both operands are true.</td>
#             <td><code>boolean isTrue = (5 &gt; 3) &amp;&amp; (7 &lt; 10); // isTrue is true</code></td>
#         </tr>
#         <tr>
#             <td>||</td>
#             <td>Logical OR operator returns true if either of the operands is true.</td>
#             <td><code>boolean isTrue = (5 > 3) || (7 > 10); // isTrue is true</code></td>
#         </tr>
#         <tr>
#             <td>! </td>
#             <td>Logical NOT operator returns the opposite boolean value of the operand.</td>
#             <td><code>boolean isFalse = !(5 > 3); // isFalse is false</code></td>
#         </tr>
#     </tbody>
# </table>

# #### **Bitwise Operators**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>&amp;</td>
#             <td>Bitwise AND operator performs a bitwise AND operation on two operands.</td>
#             <td><code>int result = 5 &amp; 3; // result is 1</code></td>
#         </tr>
#         <tr>
#             <td>|</td>
#             <td>Bitwise OR operator performs a bitwise OR operation on two operands.</td>
#             <td><code>int result = 5 | 3; // result is 7</code></td>
#         </tr>
#         <tr>
#             <td>^</td>
#             <td>Bitwise XOR operator performs a bitwise XOR operation on two operands.</td>
#             <td><code>int result = 5 ^ 3; // result is 6</code></td>
#         </tr>
#         <tr>
#             <td>~</td>
#             <td>Bitwise NOT operator performs a bitwise complement operation on the operand.</td>
#             <td><code>int result = ~5; // result is -6</code></td>
#         </tr>
#         <tr>
#             <td>&lt;&lt;</td>
#             <td>Bitwise left shift operator shifts the bits of the left operand to the left by the number of bits
#                 specified by the right operand.</td>
#             <td><code>int result = 5 &lt;&lt; 2; // result is 20</code></td>
#         </tr>
#         <tr>
#             <td>&gt;&gt;</td>
#             <td>Bitwise right shift operator shifts the bits of the left operand to the right by the number of bits
#                 specified by the right operand.</td>
#             <td><code>int result = 5 &gt;&gt; 2; // result is 1</code></td>
#         </tr>
#         <tr>
#             <td>&gt;&gt;&gt;</td>
#             <td>Bitwise unsigned right shift operator shifts the bits of the left operand to the right by the number of
#                 bits specified by the right operand and fills the leftmost bits with 0.</td>
#             <td><code>int result = -5 &gt;&gt;&gt; 2; // result is 1073741822</code></td>
#         </tr>
#     </tbody>
# </table>

# #### **Ternary Operator**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>? :</td>
#             <td>Ternary operator evaluates a boolean expression and returns one of two possible values, depending on
#                 whether the expression is true or false.</td>
#             <td><code>int max = (5 &gt; 3) ? 5 : 3; // max is 5 if the expression is true, otherwise max is 3</code></td>
#         </tr>
#     </tbody>
# </table>

# #### **Unary Operators**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Operator</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>+</td>
#             <td>Unary plus operator indicates positive value.</td>
#             <td><code>int x = +5; // x is 5</code></td>
#         </tr>
#         <tr>
#             <td>-</td>
#             <td>Unary minus operator negates the value of the operand.</td>
#             <td><code>int x = -5; // x is -5</code></td>
#         </tr>
#         <tr>
#             <td>++</td>
#             <td>Increment operator increments the value of the operand by 1.</td>
#             <td><code>int x = 5; x++; // x is now 6</code></td>
#         </tr>
#         <tr>
#             <td>--</td>
#             <td>Decrement operator decrements the value of the operand by 1.</td>
#             <td><code>int x = 5; x--; // x is now 4</code></td>
#         </tr>
#         <tr>
#             <td>!</td>
#             <td>Logical NOT operator returns the opposite boolean value of the operand.</td>
#             <td><code>boolean isFalse = !(5 &gt; 3); // isFalse is false</code></td>
#         </tr>
#         <tr>
#             <td>~</td>
#             <td>Bitwise NOT operator performs a bitwise complement operation on the operand.</td>
#             <td><code>int result = ~5; // result is -6</code></td>
#         </tr>
#         <tr>
#             <td>(type)</td>
#             <td>Cast operator converts one data type to another.</td>
#             <td><code>double d = 3.14; int i = (int) d; // i is 3</code></td>
#         </tr>
#     </tbody>
# </table>

# ## Control Structures

# #### **Conditional Statements**

# <table border=1>
#   <tr>
#     <th>Statement</th>
#     <th>Description</th>
#     <th>Example</th>
#   </tr>
#   <tr>
#     <td>if</td>
#     <td>Executes a block of code if the specified condition is true</td>
#     <td><code>if (x > y) { System.out.println("x is greater than y"); }</code></td>
#   </tr>
#   <tr>
#     <td>else</td>
#     <td>Executes a block of code if the preceding if statement is false</td>
#     <td><code>if (x > y) { System.out.println("x is greater than y"); } else { System.out.println("y is greater than or equal to x"); }</code></td>
#   </tr>
#   <tr>
#     <td>else if</td>
#     <td>Executes a block of code if the preceding if statement is false and the specified condition is true</td>
#     <td><code>if (x > y) { System.out.println("x is greater than y"); } else if (x < y) { System.out.println("x is less than y"); } else { System.out.println("x is equal to y"); }</code></td>
#   </tr>
#   <tr>
#     <td>switch</td>
#     <td>Executes a block of code based on the value of a variable</td>
#     <td><code>switch (day) { case 1: System.out.println("Monday"); break; case 2: System.out.println("Tuesday"); break; default: System.out.println("Invalid day"); }</code></td>
#   </tr>
# </table>
# 

# #### **Loop Statements**

# <table border=1>
#   <thead>
#     <tr>
#       <th>Statement</th>
#       <th>Description</th>
#       <th>Example</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>for</td>
#       <td>Repeats a block of code a fixed number of times</td>
#       <td><code>for (int i = 0; i &lt; 5; i++) { System.out.println(i); }</code></td>
#     </tr>
#     <tr>
#       <td>while</td>
#       <td>Repeats a block of code while a specified condition is true</td>
#       <td><code>int i = 0; while (i &lt; 5) { System.out.println(i); i++; }</code></td>
#     </tr>
#     <tr>
#       <td>do-while</td>
#       <td>Repeats a block of code while a specified condition is true, at least once</td>
#       <td><code>int i = 0; do { System.out.println(i); i++; } while (i &lt; 5);</code></td>
#     </tr>
#     <tr>
#       <td>foreach</td>
#       <td>Repeats a block of code for each element in an array or collection</td>
#       <td><code>int[] numbers = {1, 2, 3, 4, 5}; for (int number : numbers) { System.out.println(number); }</code></td>
#     </tr>
# </table>

# #### **Jump Statements**

# <table border=1>
#   <thead>
#     <tr>
#       <th>Statement</th>
#       <th>Description</th>
#       <th>Example</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>break</td>
#       <td>Terminates the execution of a loop or switch statement and transfers control to the statement immediately following the loop or switch</td>
#       <td><code>for (int i = 0; i &lt; 10; i++) { if (i == 5) { break; } System.out.println(i); }</code></td>
#     </tr>
#     <tr>
#       <td>continue</td>
#       <td>Skips the current iteration of a loop and continues with the next iteration</td>
#       <td><code>for (int i = 0; i &lt; 10; i++) { if (i % 2 == 0) { continue; } System.out.println(i); }</code></td>
#     </tr>
#     <tr>
#       <td>return</td>
#       <td>Terminates the execution of a method and returns a value to the caller</td>
#       <td><code>public int sum(int a, int b) { return a + b; }</code></td>
#     </tr>
#   </tbody>
# </table>

# #### **Exception Handling**

# <table border=1>
#   <thead>
#     <tr>
#       <th>Syntax</th>
#       <th>Description</th>
#       <th>Example</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>try-catch</td>
#       <td>Used to catch exceptions that may occur within a block of code</td>
#       <td>
#         <pre>
# try {
#     // code that may throw an exception
# } catch (ExceptionType1 e1) {
#     // code to handle exception of type ExceptionType1
# } catch (ExceptionType2 e2) {
#     // code to handle exception of type ExceptionType2
# } finally {
#     // code that will always execute, regardless of whether an exception is thrown or not
# }
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>throw</td>
#       <td>Used to manually throw an exception</td>
#       <td><code>if (x &lt; 0) { throw new IllegalArgumentException("x must be non-negative"); }</code></td>
#     </tr>
#     <tr>
#       <td>throws</td>
#       <td>Used to declare that a method may throw an exception</td>
#       <td><code>public void readFile(String filename) throws IOException { // code to read a file }</code></td>
#     </tr>
#   </tbody>
# </table>

# ## Data Structures

# #### **Java data Structures**

# <table border=1>
#   <thead>
#     <tr>
#       <th>Data Structure</th>
#       <th>Description</th>
#       <th>Example</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>Array</td>
#       <td>A fixed-size container that stores a sequence of elements of the same type</td>
#       <td><code>int[] numbers = new int[]{1, 2, 3, 4, 5};</code></td>
#     </tr>
#     <tr>
#       <td>ArrayList</td>
#       <td>A dynamic-size container that stores a sequence of elements of the same type</td>
#       <td>
#         <pre>
# ArrayList<String> names = new ArrayList<String>();
# names.add("Alice");
# names.add("Bob");
# names.add("Charlie");
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>LinkedList</td>
#       <td>A dynamic-size container that stores a sequence of elements of the same type, implemented as a linked list</td>
#       <td>
#         <pre>
# LinkedList<Integer> numbers = new LinkedList<Integer>();
# numbers.add(1);
# numbers.add(2);
# numbers.add(3);
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>Stack</td>
#       <td>A container that stores elements in a last-in, first-out (LIFO) order</td>
#       <td>
#         <pre>
# Stack<String> stack = new Stack<String>();
# stack.push("Alice");
# stack.push("Bob");
# stack.push("Charlie");
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>Queue</td>
#       <td>A container that stores elements in a first-in, first-out (FIFO) order</td>
#       <td>
#         <pre>
# Queue<String> queue = new LinkedList<String>();
# queue.add("Alice");
# queue.add("Bob");
# queue.add("Charlie");
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>HashMap</td>
#       <td>A container that stores key-value pairs, allowing for fast lookup of values by key</td>
#       <td>
#         <pre>
# HashMap<String, Integer> ages = new HashMap<String, Integer>();
# ages.put("Alice", 25);
# ages.put("Bob", 30);
# ages.put("Charlie", 35);
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>HashSet</td>
#       <td>A container that stores a set of unique elements</td>
#       <td>
#         <pre>
# HashSet<String> names = new HashSet<String>();
# names.add("Alice");
# names.add("Bob");
# names.add("Charlie");
#         </pre>
#       </td>
#     </tr>
#   </tbody>
# </table>

# #### **Array Operations**

# <table border=1>
#   <thead>
#     <tr>
#       <th>Operation</th>
#       <th>Description</th>
#       <th>Example</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>Creating an array</td>
#       <td>Declare and initialize an array with a fixed size</td>
#       <td><code>int[] numbers = new int[5];</code></td>
#     </tr>
#     <tr>
#       <td>Initializing an array</td>
#       <td>Assign values to each element of an array</td>
#       <td><code>int[] numbers = {1, 2, 3, 4, 5};</code></td>
#     </tr>
#     <tr>
#       <td>Accessing an array element</td>
#       <td>Get the value of an element at a given index</td>
#       <td><code>int x = numbers[2]; // x = 3</code></td>
#     </tr>
#     <tr>
#       <td>Updating an array element</td>
#       <td>Change the value of an element at a given index</td>
#       <td><code>numbers[2] = 10; // numbers = {1, 2, 10, 4, 5}</code></td>
#     </tr>
#     <tr>
#       <td>Iterating over an array</td>
#       <td>Loop through all elements of an array</td>
#       <td>
#         <pre>
# for (int i = 0; i &lt; numbers.length; i++) {
#   int x = numbers[i];
#   // do something with x
# }
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>Finding the length of an array</td>
#       <td>Get the number of elements in an array</td>
#       <td><code>int length = numbers.length; // length = 5</code></td>
#     </tr>
#     <tr>
#       <td>Sorting an array</td>
#       <td>Sort the elements of an array in ascending order</td>
#       <td>
#         <pre>
# Arrays.sort(numbers);
# // numbers = {1, 2, 4, 5, 10}
#         </pre>
#       </td>
#     </tr>
#     <tr>
#       <td>Copying an array</td>
#       <td>Create a new array with the same values as an existing array</td>
#       <td><code>int[] copy = Arrays.copyOf(numbers, numbers.length);</code></td>
#     </tr>
#     <tr>
#       <td>Searching an array</td>
#       <td>Find the index of a specific element in an array</td>
#       <td>
#         <pre>
# int index = Arrays.binarySearch(numbers, 4);
# // index = 2
#         </pre>
#       </td>
#     </tr>
#   </tbody>
# </table>

# ## Scope Level

# #### **Level of Visibility**

# <table border=1>
#     <thead>
#         <tr>
#             <th>Scope level</th>
#             <th>Keyword used</th>
#             <th>Accessible from</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>Class-level scope</td>
#             <td>static</td>
#             <td>All instances of the class and other classes in the same package</td>
#         </tr>
#         <tr>
#             <td>Instance-level scope</td>
#             <td>Not static</td>
#             <td>Instances of the class</td>
#         </tr>
#         <tr>
#             <td>Method-level scope</td>
#             <td>Not applicable</td>
#             <td>Within the method</td>
#         </tr>
#         <tr>
#             <td>Local scope</td>
#             <td>Not applicable</td>
#             <td>Within the code block (e.g., loop, conditional statement)</td>
#         </tr>
#     </tbody>
# </table>

# Note: Package-level scope, which allows access to variables and methods across classes within the same package
# without requiring explicit qualification or namespace prefix, is based on the default visibility of the class (which is package-private) rather than on any keyword or modifier, and thus is not included in this table.

# #### **Modifiers**

# Java modifiers are keywords used to define the scope, access, and behavior of classes, methods, variables, and other program components. The following table categorizes Java modifiers according to their function and syntax format:

# <table border=1>
#     <thead>
#         <tr>
#             <th>Modifier Type</th>
#             <th>Function</th>
#             <th>Syntax Format</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>Access Modifiers</td>
#             <td>Determines the accessibility of classes, methods, and variables from other classes and packages.</td>
#             <td>public, protected, private</td>
#             <td><code>public class Example { ... }<br><br>protected void method() { ... }<br><br>private int var;</code>
#             </td>
#         </tr>
#         <tr>
#             <td>Non-Access Modifiers</td>
#             <td>Modifiers that change the behavior of classes, methods, and variables without affecting accessibility.
#             </td>
#             <td>static, final, abstract, synchronized, native</td>
#             <td><code>static int count = 0;<br><br>final double PI = 3.14;<br><br>abstract void draw();<br><br>synchronized void update() { ... }<br><br>native void init();</code>
#             </td>
#         </tr>
#         <tr>
#             <td>Access and Non-Access Modifiers</td>
#             <td>Modifiers that combine both accessibility and behavior modification.</td>
#             <td>strictfp, transient, volatile</td>
#             <td><code>public strictfp class Example { ... }<br><br>transient int tempVar;<br><br>volatile boolean flag;</code>
#             </td>
#         </tr>
#     </tbody>
# </table>

# Let's discuss each of these modifier types in more detail:
# 
# ***Access Modifiers***
# 
# Access modifiers determine the accessibility of classes, methods, and variables from other classes and packages. There are three access modifiers in Java:
# 
# - public: A public class, method, or variable can be accessed from any other class in any package.
# - protected: A protected class, method, or variable can be accessed from the same package or subclass.
# - private: A private class, method, or variable can be accessed only within the same class.
# 
# ***Non-Access Modifiers***
# 
# Non-access modifiers change the behavior of classes, methods, and variables without affecting accessibility. There are five non-access modifiers in Java:
# 
# - static: A static variable or method belongs to the class rather than an instance of the class.
# - final: A final variable or method cannot be changed or overridden.
# - abstract: An abstract class or method provides a template for subclasses to implement.
# - synchronized: A synchronized method can be accessed by only one thread at a time.
# - native: A native method is implemented in a language other than Java.
# 
# ***Access and Non-Access Modifiers***
# 
# These modifiers combine both accessibility and behavior modification. There are two modifiers in this category:
# 
# - strictfp: A strictfp class or method ensures consistent floating-point calculations across different platforms.
# - transient: A transient variable is not saved when the object is serialized.
# - volatile: A volatile variable is accessed by multiple threads, and its value is always up-to-date.

# ## Anatomy of Methods in Java

# <table border=1>
#     <thead>
#         <tr>
#             <th>Component</th>
#             <th>Description</th>
#             <th>Example</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>Access Modifier</td>
#             <td>Determines the access level of the method, such as public, private, or protected.</td>
#             <td><code>public</code></td>
#         </tr>
#         <tr>
#             <td>Return Type</td>
#             <td>Specifies the data type of the value that the method will return, such as <code>int</code>,
#                 <code>String</code>, or <code>void</code>.
#             </td>
#             <td><code>int[]</code></td>
#         </tr>
#         <tr>
#             <td>Method Name</td>
#             <td>Defines the name of the method, which should be descriptive of its purpose.</td>
#             <td><code>filterEvenNumbers</code></td>
#         </tr>
#         <tr>
#             <td>Parameters</td>
#             <td>Specifies the input parameters that the method expects to receive, including their data types and names.
#             </td>
#             <td><code>(int[] numbers)</code></td>
#         </tr>
#         <tr>
#             <td>Method Body</td>
#             <td>Contains the code that the method will execute when it is called, enclosed in curly braces.</td>
#             <td><code>{ List&lt;Integer&gt; evenNumbers = new ArrayList&lt;&gt;(); for (int num : numbers) { if (num % 2 == 0) { evenNumbers.add(num); } } int[] result = evenNumbers.stream().mapToInt(Integer::intValue).toArray(); return result; }</code>
#             </td>
#         </tr>
#     </tbody>
# </table>

# Example:

# In[ ]:


public int[] filterEvenNumbers(int[] numbers) {
    List<Integer> evenNumbers = new ArrayList<>();
    for (int num : numbers) {
        if (num % 2 == 0) {
            evenNumbers.add(num);
        }
    }
    int[] result = evenNumbers.stream().mapToInt(Integer::intValue).toArray();
    return result;
}


# ## OOP with Java

# <table>
#     <thead>
#         <tr>
#             <th>Concept</th>
#             <th>Description</th>
#             <th>Example Syntax</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td rowspan="2">Class</td>
#             <td>Blueprint for creating objects. Defines the properties and methods that an object will have.</td>
#             <td>
#                 <pre lang="java">public class MyClass {
#     private int myVar;
#   public void setMyVar(int var) {
#   myVar = var;
#   }
#   }</pre>
#             </td>
#         </tr>
#         <tr>
#             <td>Can be instantiated to create objects. Objects have their own instance variables and can call the
#                 methods of their class.</td>
#             <td>
#                 <pre lang="java">MyClass obj = new MyClass();
#   obj.setMyVar(5);</pre>
#             </td>
#         </tr>
#         <tr>
#             <td>Inheritance</td>
#             <td>Allows a class to inherit properties and methods from another class. The inherited class is called the
#                 superclass or parent class, and the inheriting class is called the subclass or child class.</td>
#             <td>
#                 <pre lang="java">public class Vehicle {
#   protected int numWheels;
#   public void setNumWheels(int num) {
#   numWheels = num;
#   }
#   }
#   
#   public class Car extends Vehicle {
#   private String model;
#   public void setModel(String mod) {
#   model = mod;
#   }
#   }</pre>
#             </td>
#         </tr>
#         <tr>
#             <td>Polymorphism</td>
#             <td>The ability of an object to take on multiple forms. In Java, this is often achieved through method
#                 overriding and interfaces.</td>
#             <td>
#                 <pre lang="java">public interface Animal {
#   public void makeSound();
#   }
#   
#   public class Dog implements Animal {
#   public void makeSound() {
#   System.out.println("Woof");
#   }
#   }
#   
#   public class Cat implements Animal {
#   public void makeSound() {
#   System.out.println("Meow");
#   }
#   }</pre>
#             </td>
#         </tr>
#         <tr>
#             <td>Encapsulation</td>
#             <td>The practice of keeping instance variables private and providing public methods to access or modify
#                 them. This helps to prevent unintended modification of object state and maintain data integrity.</td>
#             <td>
#                 <pre lang="java">public class BankAccount {
#   private int balance;
#   
#   public void deposit(int amount) {
#   balance += amount;
#   }
#   
#   public void withdraw(int amount) {
#   if (balance >= amount) {
#   balance -= amount;
#   } else {
#   System.out.println("Insufficient funds");
#   }
#   }
#   
#   public int getBalance() {
#   return balance;
#   }
#   }</pre>
#             </td>
#         </tr>
#         <tr>
#             <td>Abstraction</td>
#             <td>The process of hiding implementation details and providing a simplified view of an object to its users.
#                 Abstract classes and interfaces are often used to achieve abstraction.</td>
#             <td>
#                 <pre lang="java">public abstract class Shape {
#   protected int x, y;
#   public void setPosition(int x, int y) {
#   this.x = x;
#   this.y = y;
#   }
#   public abstract double getArea();
#   }
#   
#   public class Circle extends Shape {
#   protected int radius;
#   public void setRadius(int r) {
#   radius = r;
#   }
#   public double getArea() {
#   return Math.PI * radius * radius;
#   }
#   }</pre>
#             </td>
#         </tr>
#     </tbody>
# </table>
