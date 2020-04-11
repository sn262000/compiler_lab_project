%{ 
	#include<stdio.h>
	#include<string.h>
	int n, w, c; 
%} 
%% 
\n 	{ n++; } 
[^ \n\t]+  { 	w++; 
				c = c + yyleng; 
			} 
.c++; 
%% 
int yywrap(void) 
{ 
	return 1; 
} 


main(int argc, char *argv[]) 
{ 
	char p[20];
	extern FILE* yyin;
	strcpy(p,argv[1]);
	yyin = fopen(p, "r"); 
	yylex(); 
	printf("\nLine= %d\tword=%d\ttotal char=%d \n", n, w, c); 
} 
