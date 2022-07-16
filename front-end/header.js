$(document).ready(function () {
    $(".toggle-nav").click(function () {
        toggleNav();
    });

    function toggleNav() {
        if ($(".drawer").hasClass("show-nav")) {
            $(".drawer").removeClass("show-nav");
            $(".nav-icon").removeClass("open");
            $("nav li.gotsMenus").removeClass("showSub");
        } else {
            $(".drawer").addClass("show-nav");
            $(".nav-icon").addClass("open");
            $("nav li.gotsMenus").removeClass("showSub");
        }
    }
    //add (nav li li) class to below function if you want to close dropdown when inside link is clicked
    $(".closeButton").click(function () {
        $(".drawer").removeClass("show-nav");
        $(".nav-icon").removeClass("open");
        $("nav li.gotsMenus").removeClass("showSub");
    });

    const navLink = document.querySelectorAll(".header-nav-item");

    function linkAction() {
        var slashpath = window.location.pathname;
        if (slashpath[slashpath.length - 1] === "/") {
            slashpath = slashpath.substring(slashpath.length - 1);
        }
        navLink.forEach((n) => {
            if (slashpath === n.getAttribute("path")) {
                n.classList.add("active-link");
            }
        });
    }

    linkAction();
});
