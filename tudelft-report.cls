% This template aims to simplify and improve the (Xe)LaTeX template provided
% by the TU Delft. Original template by TU Delft. Rewritten template by Daan
% Zwaneveld (https://dzwaneveld.github.io) and subsequently 
% adapted by Philip Soliman (https://github.com/PhilipSoliman).

%% ----------------------------------------------------------------------
%%    Setting up the class, main packages and basic definitions
%% ----------------------------------------------------------------------
%% Class is based on the default book class and options will be passed
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tudelft-report}[05-09-2024 v1.7.1 TU Delft Report Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[10pt,oneside]{book}

%% Main packages in the document --- Some are imported later in the class file
\RequirePackage{mathtools}  % Mathematical tools to use with amsmath
\RequirePackage{amsthm}     % Theorem environments
\RequirePackage{amssymb}    % Extended symbol collection
\RequirePackage{siunitx}    % Comprehensive (SI) units package

\RequirePackage{tabularx}   % Tabulars with adjustable-width columns
\RequirePackage{booktabs}   % Publication quality tables
\RequirePackage{longtable}  % Allow tables to flow over page boundaries
\RequirePackage{multirow}   % Create tabular cells spanning multiple rows

\RequirePackage{graphicx}   % Enhanced support for images
\RequirePackage{float}      % Improved interface for floating objects
\RequirePackage[labelfont=bf,justification=centering,footnotesize]{caption} % Captions
\RequirePackage{subcaption} % Support for sub-captions
\RequirePackage{pdfpages}   % Include PDF documents

\RequirePackage[pdfusetitle,hidelinks]{hyperref} % Extensive support for hypertext
\RequirePackage[noabbrev]{cleveref} % Intelligent cross-referencing
\RequirePackage{xcolor}     % Driver-independent color extensions
\RequirePackage{tikz}       % Create PostScript and PDF graphics
\RequirePackage{xspace}     % Define commands that appear not to eat spaces
\RequirePackage{microtype}  % Refinements towards typographical perfection

\RequirePackage{geometry}   % Customize document dimensions
\RequirePackage{titlesec}   % Select alternative section titles
\RequirePackage{titletoc}   % Alternative headings for toc
\RequirePackage{fancyhdr}   % Control of page headers and footers
\RequirePackage{enumitem}   % Control layout of itemize, enumerate, description
\RequirePackage{etoolbox}   % Toolbox of programming facilities
\RequirePackage{iftex}      % Adds if-else statements to support multiple compilers
\RequirePackage{datetime}   % Change format of \today

\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{algorithmicx}
\RequirePackage{tikz}
\RequirePackage{tikz-cd}
\RequirePackage{thmtools}
\RequirePackage[framemethod=TikZ]{mdframed}
\RequirePackage{listings}
\RequirePackage{afterpage}

%% Establish commands for the subtitle, subject, affiliation, cover image and table of authors
\newcommand*{\subtitle}[1]{\def\@subtitle{#1}}
\newcommand*{\subject}[1]{\def\@subject{#1}}
\newcommand*{\affiliation}[1]{\def\@affiliation{#1}}
\newcommand*{\coverimage}[1]{\def\@coverimage{#1}}
\newcommand*{\covertable}[1]{\def\@covertable{#1}}
\newcommand*{\logoimage}[1]{\def\@logoimage{#1}}
\newcommand*{\defenseDate}[1]{\def\@defenseDate{#1}}
\newcommand*{\defenseTime}[1]{\def\@defenseTime{#1}}
\newcommand*{\publishDate}[1]{\def\@publishDate{#1}}


%% Scale the margins to be slightly smaller than default (.7)
\geometry{a4paper,hscale=0.75,vscale=0.8}

% -----------------------------------------------------------------------
%     Setting up the colors
% -----------------------------------------------------------------------
% cover page
\definecolor{title}{HTML}{945357} % 
\definecolor{chapter}{HTML}{945357} % Deep Red
\definecolor{subtitle}{HTML}{FFFFFF} % White
\definecolor{subject}{HTML}{FFFFFF} % White
\definecolor{author}{HTML}{FFFFFF} % White
\definecolor{affiliation}{HTML}{FFFFFF} % White
\definecolor{coverbackground}{HTML}{253242} % Dark Navy

% chapters and sections
\definecolor{chapter}{HTML}{945357} % Deep Red
\definecolor{section}{HTML}{945357} % Deep Red
\definecolor{subsection}{HTML}{253242} % Dark Navy
\definecolor{subsubsection}{HTML}{869099} % Muted Blue-Gray

% header and footer
\definecolor{header}{HTML}{253242} % Dark Navy

% hyperlinks
\definecolor{link}{HTML}{1A2533} % Very Dark Blue
\definecolor{file}{HTML}{7EAFF1} % Soft Sky Blue
\definecolor{cite}{HTML}{651319} % Dark Red
\definecolor{url}{HTML}{253242} % Dark Navy
\definecolor{caption}{HTML}{7A8F99} % Muted Blue-Gray

% equations
\definecolor{eqnnumber}{HTML}{565C68} % Grayish Blue

% code listings
\definecolor{bgcolor}{HTML}{E0E0E0} % Light Gray
\definecolor{commentcolor}{HTML}{869099} % Muted Blue-Gray
\definecolor{keywordcolor}{HTML}{7EAFF1} % Soft Sky Blue
\definecolor{stringcolor}{HTML}{B38B6D} % Golden Brown
\definecolor{identifiercolor}{HTML}{E8F0F3} % Very Light Blue
\definecolor{numbercolor}{HTML}{945357} % Deep Reddish Brown

%% ----------------------------------------------------------------------
%%    Setting up the fonts
%% ----------------------------------------------------------------------
\ifPDFTeX
    %% With pdfLaTeX, use Paletino as the main font and Roboto Slab as title fonts
    \RequirePackage[T1]{fontenc}  % Select T1 font encoding
    \RequirePackage{newpxtext}    % Palatino-like font...
    \RequirePackage{newpxmath}    % ...with support for mathematics

    \newcommand{\chapterstyle}{\fontfamily{RobotoSlab-TLF}\fontseries{light}\selectfont}
    \newcommand{\largechapterstyle}{\fontfamily{RobotoSlab-TLF}\fontseries{thin}\selectfont}
\else
    %% If XeLaTeX or LuaLaTeX is set as the compiler, the TU Delft house style fonts are used (see https://www.tudelft.nl/huisstijl/bouwstenen/typografie).
    \RequirePackage[no-math]{fontspec} % Advanced font selection

    %% Use Arial as the main font
    \setmainfont{Tex Gyre Heros}
    \setmathsf{Tex Gyre Heros}
    \setmathtt{Tex Gyre Heros}
    \setmathrm{Tex Gyre Heros}

    %% Use Nimbus Sans L as the cover font
    \newfontfamily\largecoverstyle{Nimbus Sans L}
    \newfontfamily\coverstyle{Nimbus Sans L}[Weight=Light]

    % Use Roboto Slab as the chapter font
    \newfontfamily\largechapterstyle{Roboto Slab Thin}
    \newfontfamily\chapterstyle{Roboto Slab Light}
    \newfontfamily\sectionstyle{Roboto Slab}[Weight=400]
    \newfontfamily\subsectionstyle{Roboto Slab}[Weight=400]
    \newfontfamily\subsubsectionstyle{Roboto Slab}[Weight=300]

    % helvetica alternative (requires installation of the font)
    % \newfontfamily\largechapterstyle{Helvetica Light}
    % \newfontfamily\chapterstyle{Helvetica Light}
    % \newfontfamily\sectionstyle{Helvetica}
    % \newfontfamily\subsectionstyle{Helvetica}
    % \newfontfamily\subsubsectionstyle{Helvetica Light}
\fi

%% ----------------------------------------------------------------------
%%    Formatting the titles and table of contents
%% ----------------------------------------------------------------------
\makeatletter
\let\@chapterimage\relax
\newcommand\chapterimage[1]{%
  \if\relax\detokenize{#1}\relax
  \else
    \gdef\@chapterimage{\centering\vspace{8cm}\smash{\includegraphics[width=0.5\textwidth,keepaspectratio]{#1}}}
  \fi
}

\newcommand\nochapterimage{
  \let\@chapterimage\relax
}

%% Format the chapter titles and spacing
\titleformat{\chapter}[display]
    {\flushright}
    {\fontsize{96}{96}\selectfont\largechapterstyle\color{chapter}\thechapter}
    {0pt}
    {\Huge\chapterstyle\color{chapter}}
    [\@chapterimage]

\titlespacing*{\chapter}{0pt}{0pt}{2\baselineskip}

%% Format the section titles and spacing
\titleformat{\section}
    {\Large\sectionstyle\bfseries\color{section}}
    {\thesection.}
    {5pt}
    {}
\titlespacing*{\section}{0pt}{\baselineskip}{0pt}

%% Format the subsections titles and spacing
\titleformat{\subsection}
    {\large\subsectionstyle\bfseries\color{subsection}}
    {\thesubsection.}
    {5pt}
    {}
\titlespacing*{\subsection}{0pt}{\baselineskip}{0pt}

%% Format the subsubsections titles and spacing
\titleformat{\subsubsection}
    {\subsubsectionstyle\bfseries\color{subsubsection}}
    {\thesubsubsection.}
    {5pt}
    {}
\titlespacing*{\subsubsection}{0pt}{\bigskipamount}{0pt}

%% Reduce the vertical white space between chapters in the table of contents
\dottedcontents{chapter}[1.5em]{\vspace{0.5\baselineskip}\bfseries}{1.5em}{0pt}

%% ----------------------------------------------------------------------
%%    Formatting the header and footer
%% ----------------------------------------------------------------------
%% redefine fancyhdr commands
\renewcommand{\chaptermark}[1]{\markboth{\textbf{\thechapter} #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\S\arabic{section} \ #1}}

%% Format the header and footer of 'plain' pages
\fancypagestyle{plain}{%
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\headrule}{\hbox to\headwidth{%
    \color{header}\leaders\hrule height \headrulewidth\hfill}}
    \fancyfoot[C]{\chapterstyle\color{header}\thepage}}

%% set default style
\pagestyle{fancy} % Set the package defaults and the additional changes as the style

%% Format the header and footer of 'fancy' pages (based on twoside option)
\if@twoside
    \fancyhf{}
    \fancyhead[LE,RO]{\chapterstyle\color{header}\thepage}
    \fancyhead[RE]{\chapterstyle\color{header}\nouppercase{\leftmark}}
    \fancyhead[LO]{\chapterstyle\color{header}\nouppercase{\rightmark}}
    \RequirePackage{emptypage} % Remove header and footer on empty pages
\else 
    \fancyhf{}
    \fancyhead[L]{\chapterstyle\color{header}\nouppercase{\leftmark}}%
    \fancyhead[C]{\chapterstyle\color{header}\nouppercase{\rightmark}}%
    \fancyhead[R]{\chapterstyle\color{header}\thepage}
\fi

%% ----------------------------------------------------------------------
%%    Setting up the \makecover command for the cover page
%% ----------------------------------------------------------------------
\newcommand*{\makecover}{
    %% Use the Tikz library positioning and clear the page header and footer
    \usetikzlibrary{positioning}
    \thispagestyle{empty}

    %% Construct the cover page with Tikz
    \begin{tikzpicture}[overlay,remember picture]

        %% Add the cover image
        \ifdefined\@coverimage
            \node[above=0,inner sep=0] at (current page.south) {%
                \includegraphics[width=\paperwidth]{\@coverimage} 
            };
        \else
            \pagecolor{coverbackground}\afterpage{\nopagecolor}
        \fi

        %% Add the affiliation on the left
        \node[rotate=90,below right=40mm and 3mm] at (current page.west) {%
            \ifdefvoid{\@affiliation}{}{\color{affiliation}\coverstyle\@affiliation}};

        %% Add the logo in the bottom left
        \node[above right=10mm] at (current page.south west) {%
            \includegraphics[width=0.35\linewidth]{\@logoimage}};

        %% Add the banner with the title, subtitle, subject and author(s)
        \node[below=2cm,fill=black!85,minimum width={\paperwidth},inner ysep=25pt,opacity=0.6,text opacity=1] at (current page.north) {%
            \begin{minipage}{0.9\paperwidth}
                %% Format and add the title
                \color{title}\raggedright\largecoverstyle\fontsize{40}{40}\selectfont%
                    \@title \\[0.5ex]
                %% Format and add (optional) subtitle
                \coverstyle\fontsize{24}{24}\selectfont%
                \ifdefvoid{\@subtitle}{}{\color{subtitle}\@subtitle \\[2.5ex]}%
                %% Format and add (optional) subject
                \coverstyle\fontsize{20}{20}\selectfont%
                \ifdefvoid{\@subject}{}{\color{subject}\@subject \\[0.5ex]}
                %% Format and add author or table of authors
                \coverstyle\fontsize{20}{20}\selectfont%
                    \ifdefvoid{\@covertable}{\color{author}\@author}{\@covertable}
            \end{minipage}};

    \end{tikzpicture}
    \newpage
}

%% ----------------------------------------------------------------------
%%    Formatting equations
%% ----------------------------------------------------------------------
\numberwithin{equation}{chapter}
\AtBeginDocument{
    \renewcommand{\tagform@}[1]{\textcolor{eqnnumber}{\textrm{(\ignorespaces#1\unskip)}}}
    \renewcommand{\theequation}{\textcolor{eqnnumber}{\textrm{\arabic{chapter}.\arabic{equation}}}}
}

%% ----------------------------------------------------------------------
%%    Setting up listings
%% ----------------------------------------------------------------------
\lstdefinestyle{mystyle}{
  backgroundcolor=\color{bgcolor},   
  basicstyle=\ttfamily\scriptsize\color{identifiercolor}, % Default text
  keywordstyle=\color{keywordcolor}\bfseries, % Keywords
  commentstyle=\color{commentcolor}\itshape, % Comments
  stringstyle=\color{stringcolor}, % Strings
  numberstyle=\footnotesize\color{numbercolor}, % Numbers
  breakatwhitespace=false,         
  showstringspaces=false,
  breaklines=true,
  frame=single,
  rulecolor=\color{bgcolor},
  rulesep=0pt,
  captionpos=b,
  numbers=left,
  showtabs=false,
  showspaces=false,
  numbersep=5pt,
  tabsize=2,
}
\lstset{style=mystyle}

%% ----------------------------------------------------------------------`
%%    Setting up the Hyperlinks
%% ----------------------------------------------------------------------
\hypersetup{
    colorlinks=true,
    linkcolor=link,
    filecolor=file,
    urlcolor=url,
    citecolor=cite,
    pdftitle={AM Thesis Philip Soliman},
    pdfpagemode=FullScreen,
    pdfauthor={Philip Soliman},
    pdfkeywords={Thesis}
}
\urlstyle{same}

\AtBeginDocument{%
  \renewcommand{\crefname}[3]{\textsf{\color{link}#1#2#3}} % Singular form
  \renewcommand{\Crefname}[3]{\textsf{\color{link}#1#2#3}} % Capitalized form
}
% \crefformat{default}{\textsf{\color{link}#2#1#3}}