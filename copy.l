%{
#include<stdio.h>
%}

%%
[\t ]+ fprintf(yyout," ");    
.|\n fprintf(yyout,"%s",yytext);  
%%

int yywrap()
{
return 1;
}

int main(int argc, char *argv[])
{
	char p[20];
	extern FILE* yyin;
	strcpy(p,argv[1]);
	yyin=fopen(p,"r");    
	yyout=fopen("copy.txt","w");

yylex();
return 0;
}
