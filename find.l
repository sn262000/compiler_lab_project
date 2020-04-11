%% 
#include<stdio.h>
#include<string.h>
[a-zA-Z]+ check(yytext); 
%% 

char p[20];
int main(int argc, char *argv[]) 
{ 
	printf("\nEnter word to find\n");
	strcpy(p,argv[1]);
	yylex(); 
	return 0; 
} 
void check(char *str) 
{ 
	FILE *fp; 
	char temp[30];
	fp=fopen(p, "r"); 

	while((fscanf(fp, "%s", temp))!=EOF) 
	{ 
		if(!(strcmp(temp, str))) 
		{ 
			printf("FOUND\n"); 
			return; 
		} 
		
	} 
	printf("NOT FOUND\n"); 
		return; 
} 
